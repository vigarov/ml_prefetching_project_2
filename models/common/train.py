import os
import warnings

import torch
import torch.nn as nn
import torchmetrics
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.common.build_model import build_model
import models.common.config as conf
from models.common.data_parser import *
from models.common.dataset import PageFaultDataset, causal_mask
from models.common.trained_tokenizers import special_tokenizers as st
from models.common.trained_tokenizers.special_tokenizers import SimpleTokenIdList


def greedy_decode(model,
                  source_data, source_mask,
                  tokenizer_tgt: st.TokenizerWrapper, start_stop_tokens: list,
                  max_len, device):
    assert start_stop_tokens is not None and len(start_stop_tokens) == 2
    start_idx, end_idx = tokenizer_tgt.token_to_id(start_stop_tokens[0]), tokenizer_tgt.token_to_id(
        start_stop_tokens[1])
    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source_data, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(start_idx).type_as(source_data).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_token = torch.max(prob, dim=1)  # Greedy
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source_data).fill_(next_token.item()).to(device)], dim=1
        )

        if next_token == end_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, config, validation_ds: DataLoader,
                   tokenizer_tgt: st.TokenizerWrapper, start_stop_tokens: list,
                   max_len,
                   device, print_msg, global_step, writer, num_examples=3):
    model.eval()
    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    count = 0

    source_texts = []
    expected = []
    predicted = []
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (b, I)
            encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, I)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            if config['attention_model'] == "transformer":
                model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, start_stop_tokens, max_len,
                                          device)
            elif config['attention_model'] == "retnet":
                model_out = model.custom_generate(encoder_input, max_new_tokens=conf.OUTPUT_FEATURES[0].max_len, do_sample=False, early_stopping=True)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(SimpleTokenIdList(model_out.detach().cpu().numpy().tolist()))

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the source, target and model output
            print_msg('-' * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICT:  ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-' * console_width)
                break

    if writer:
        # Evaluate the character error rate
        # Compute the char error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

        # Compute the BLEU metric
        new_pred = [set(a.split(" ")) for a in predicted]
        new_expected = [set(a.split(" ")) for a in expected]
        acc = [len(a.intersection(b)) / len(a) for a, b in zip(new_pred, new_expected)]
        acc = sum(acc) / len(acc)
        writer.add_scalar('validation accuracy', acc, global_step)
        writer.flush()


def get_tokenizers(config) -> ((st.ConcatTokenizer | list[st.TokenizerWrapper]), st.TokenizerWrapper):
    SPACE_SPLITTER = st.Splitter(lambda input_str: input_str.split(), config["list_elem_separation_token"])
    HEX_VOCAB = ["0x"] + [hex(j)[2:].zfill(2) for j in range(0xff + 1)]
    HEX_ADDRESS_SPLITTER = st.Splitter(lambda address: [address[i:i + 2] for i in range(0, len(address), 2)])
    SPECIAL_TOKENS = config["bpe_special_tokens"]

    def get_feature_tokenizer(tok_file: str, feature: conf.Feature, padder=False,
                              sentence_like_wrap_mode: str | None = None) -> st.TokenizerWrapper:
        is_list = "list" in feature.type
        primitive_feature_type = feature.get_primitive_type()

        tokenizer_path = Path(tok_file.format(primitive_feature_type))
        assert tokenizer_path.exists() and tokenizer_path.suffix == ".json"
        tokenizer = Tokenizer.from_file(tokenizer_path.absolute().as_posix())
        splitter = None
        if is_list:
            splitter = SPACE_SPLITTER
        pad_token = config["pad_token"] if padder else None
        wrap_params = None
        if sentence_like_wrap_mode is not None:
            assert sentence_like_wrap_mode in ["insert_lr", "no_insert_get_right_index"]
            wrap_params = st.WrapParameters(
                config["start_stop_generating_tokens"],
                sentence_like_wrap_mode)

        return st.TokenizerWrapper(
            tokenizer,
            len(config["bpe_special_tokens"]),
            feature.max_len,
            splitter=splitter,
            pad_token=pad_token,
            wrap_parameters=wrap_params)

    def get_feature_tokenizer_dict(tok_file, features: list[conf.Feature], all_padders=False):
        ret_dict = {}
        for feature in features:
            sentence_like_wrap_mode = None
            if feature.name == "prev_faults":
                sentence_like_wrap_mode = "insert_lr"
            ret_dict[feature.name] = get_feature_tokenizer(tok_file, feature, padder=all_padders,
                                                           sentence_like_wrap_mode=sentence_like_wrap_mode)
        return ret_dict

    def get_tok_list_from_dict(features, feature_tokenizer_dict):
        return [feature_tokenizer_dict[feature.name] for feature in features]

    token_type = config["embedding_technique"]
    tok_file = config['tokenizer_files']
    input_features, output_features = config["input_features"], config["output_features"]
    assert len(output_features) == 1 and "list" in output_features[0].type
    # To build the tokenizers, see make_tokens.py
    if token_type == "concat_tokens":
        out_tokenizer = get_feature_tokenizer(tok_file, output_features[0], padder=True,
                                              sentence_like_wrap_mode="no_insert_get_right_index")
    else:
        out_tokenizer = st.TokenizerWrapper(
            st.SimpleCustomVocabTokenizer(HEX_VOCAB, SPECIAL_TOKENS, HEX_ADDRESS_SPLITTER),
            len(SPECIAL_TOKENS),
            output_features[0].max_len,
            splitter=SPACE_SPLITTER,
            pad_token=config["pad_token"],
            wrap_parameters=st.WrapParameters(
                config["start_stop_generating_tokens"],
                "no_insert_get_right_index")
        )
    if token_type in ["concat_tokens", "hextet_concat"]:
        inp_ret_dict = get_feature_tokenizer_dict(tok_file, input_features, all_padders=False)
        if token_type == "hextet_concat":
            for feature in input_features:
                prim_feature_type = feature.get_primitive_type()
                is_list = "list" in feature.type
                splitter = None
                if is_list:
                    splitter = SPACE_SPLITTER
                if "hex" in prim_feature_type or "bit" in prim_feature_type:
                    if "hex" in prim_feature_type:
                        custom_vocab = HEX_VOCAB
                        if "number" not in prim_feature_type:
                            input_splitter = HEX_ADDRESS_SPLITTER
                        else:
                            # our input is going to be either 0x{2*N} or 0x{2*N+1}
                            # In the first case, no problem. In the second case however, we must add a zero after the 0x
                            input_splitter = st.Splitter(
                                lambda number: [number[i:i + 2] for i in range(0, len(number), 2)] if len(
                                    number) % 2 == 0
                                else [number[:2], '0' + number[2]] + [number[i:i + 2] for i in range(3, len(number), 2)]
                            )
                    else:  # "bit" in prim_feature_type
                        custom_vocab = ['0', '1']
                        input_splitter = st.Splitter(lambda bitmap: [bitmap[i] for i in range(len(bitmap))])

                    inp_ret_dict[feature.name] = st.TokenizerWrapper(
                        st.SimpleCustomVocabTokenizer(custom_vocab, SPECIAL_TOKENS, input_splitter=input_splitter),
                        len(SPECIAL_TOKENS),
                        feature.max_len,
                        splitter)
        final_input_tokenizer = st.ConcatTokenizer(
            config["feature_separation_token"],
            config["pad_token"],
            get_tok_list_from_dict(input_features, inp_ret_dict))
        return final_input_tokenizer, out_tokenizer
    elif token_type in ["meta_transformer", "embed_concat"]:
        inp_ret_dict = get_feature_tokenizer_dict(tok_file, input_features, all_padders=True)
        # Since each feature will be embedded differently, each tokenizer must be its own padder --> must recreate dict
        return get_tok_list_from_dict(input_features, inp_ret_dict), out_tokenizer
    else:
        assert token_type == "onetext"
        raise NotImplementedError
        tokenizer_path = Path(tok_file.format("onetext"))
        assert tokenizer_path.exists() and tokenizer_path.suffix == ".json"
        inp_ret_dict = Tokenizer.from_file(tokenizer_path.absolute().as_posix())
        # TODO: Use gpt model or custom build bpe, can be loaded online; c.f. tokenizers doc


def get_ds(config, generator) -> (
        DataLoader, DataLoader, (st.ConcatTokenizer | list[st.TokenizerWrapper]), st.TokenizerWrapper):
    # It only has the train split, so we divide it overselves
    if "processed" in config["data_path"]:
        df_raw = read_csv(config["data_path"])
    else:
        df_raw = process(config["data_path"], config["past_window"], config["k_predictions"],
                         save=False)  # load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    df_raw = df_raw.astype({col: str for col in df_raw.columns if df_raw[col].dtype == "int64"})
    print("loaded data")
    # Build tokenizers
    src_tokenizer, tgt_tokenizer = get_tokenizers(config)

    train_tensor_size = int(config["train_test_split"] * len(df_raw))
    indices = torch.arange(
        len(df_raw))  # torch.randperm(len(df_raw), generator=generator)  # TODO think about overlapping pfault windows

    train_ds = PageFaultDataset(config, df_raw, indices[:train_tensor_size],
                                src_tokenizer,
                                tgt_tokenizer)
    val_ds = PageFaultDataset(config, df_raw, indices[train_tensor_size:],
                              src_tokenizer,
                              tgt_tokenizer)

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer


def get_model(config, inp_tokenizer: st.ConcatTokenizer | list[st.TokenizerWrapper] | st.TokenizerWrapper,
              out_tokenizer: st.TokenizerWrapper):
    embedding_type = config["embedding_technique"]
    out_vocab_size = out_tokenizer.get_vocab_size()
    out_seq_len = int(config["output_features"][0].max_len)
    if embedding_type in ["concat_tokens", "hextet_concat"]:
        src_vocab_size = inp_tokenizer.get_vocab_size()
        used_features = config["input_features"]
        inp_seq_len = (sum([int(feature.max_len) for feature in used_features]) + len(
            used_features) - 1)  # we add a separator token between features
    elif embedding_type == "onetext":
        src_vocab_size = inp_tokenizer.get_vocab_size()
        # inp_seq_len = ? TODO
        raise NotImplementedError
    else:
        assert embedding_type in ["meta_transformer", "embed_concat"]
        # TODO: we might have to slightly adapt build_model for those two, which is why I am currently taking the tokenizers themselves as input
        # TODO cont: This might however not be needed, maybe we can do with only the different sizes/lengths
        raise NotImplementedError

    pad_token_id = inp_tokenizer.token_to_id(config["pad_token"])[0]
    eos_token_id = inp_tokenizer.token_to_id(config["start_stop_generating_tokens"][1])[0]
    model = build_model(config, src_vocab_size, out_vocab_size, inp_seq_len, out_seq_len, pad_token_id, eos_token_id)
    return model


def train_model(model):
    config = conf.get_config(model=model)
    # Define the device
    device = conf.get_device()

    # Check if we're in the middle of a "train" : if yes, load the current random sample (based off the seed)
    # Else, take (new) random seed
    was_currently_training = len(conf.source_model_files(config)) != 0
    generator = torch.Generator(device)
    mp = Path(conf.get_model_full_path(config))
    # Make sure the model's run folder exists
    mp.mkdir(parents=True, exist_ok=True)
    # Also create the generator folder
    gen_dir = mp / conf.GENERATOR_PREFIX
    gen_dir.mkdir(parents=True, exist_ok=True)
    if was_currently_training:
        seed_path = gen_dir / conf.SEED_FN
        assert seed_path.exists()
        seed = torch.load(seed_path.absolute().as_posix())
        generator = generator.manual_seed(seed)
        state_path = gen_dir / conf.STATE_FN
        assert state_path.exists()
        gen_state = torch.load(state_path.absolute().as_posix())
    else:
        torch.save(generator.seed(), (gen_dir / conf.SEED_FN).as_posix())

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config, generator)
    # Save generator TODO: useless to do here, save at weight save time iff generator impacts training
    torch.save(generator.get_state(), (gen_dir / conf.STATE_FN).as_posix())
    model = get_model(config, tokenizer_src, tokenizer_tgt).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_folder'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = conf.latest_weights_file_path(config) if preload == 'latest' \
        else conf.get_weights_file_path(config, preload) if preload \
        else None
    if model_filename:
        assert was_currently_training
        generator.set_state(
            gen_state)  # TODO: I think we might not need that - past weight initialization, there is no randomness right? Incorrect for dropout? Incorrect for beam search?
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    embedding_type = config["embedding_technique"]
    if embedding_type in ["meta_transformer", "embed_concat"]:
        # TODO: one loss per embedding layer? imo yes
        raise NotImplementedError
    else:
        input_src_potential_pad_ids = tokenizer_src.token_to_id(config["pad_token"])
        output_pad_id = tokenizer_tgt.token_to_id(config["pad_token"])
        assert len(input_src_potential_pad_ids) == 1 and input_src_potential_pad_ids[0] == output_pad_id
        loss_fn = nn.CrossEntropyLoss(ignore_index=output_pad_id,
                                      label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)  # (B, I)
            if config['attention_model'] == "transformer":
                decoder_input = batch['decoder_input'].to(device)  # (B, O')
                encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, I)
                decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, O', O')
                # Run the tensors through the encoder, decoder and the projection layer
                encoder_output = model.encode(encoder_input, encoder_mask)  # (B, I, D)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input,
                                              decoder_mask)  # (B, O', D)
                proj_output = model.project(decoder_output)  # (B, O', D)

            elif config['attention_model'] == "retnet":
                outputs = model(encoder_input)
                proj_output = outputs.get("logits")
                print(proj_output.shape)
            else:
                exit(-3)

            # Compare the output with the label
            label = batch['label'].to(device)  # (B, O')
            print(label.shape)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_tgt, config["start_stop_generating_tokens"],
                       config["output_features"][0].max_len,
                       device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = conf.get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
    return model


# def multi_config_train():
#     configs = get_all_configs()
#     for c in configs:
#         train_model(c)
# todo move to common/train


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    train_model("retnet")
