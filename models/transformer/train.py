from models.transformer.trained_tokenizers.special_tokenizers import SimpleTokenIdList

from models.transformer.model import build_model
from models.transformer.dataset import PageFaultDataset, causal_mask
from models.transformer.config import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from models.transformer.data_parser import *
import warnings
from tqdm import tqdm
import os
from pathlib import Path
from pandas import read_csv

from tokenizers import Tokenizer


from models.transformer.trained_tokenizers import special_tokenizers as st

import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def greedy_decode(model,
                  source_data_list: list[torch.Tensor],
                  source_mask,
                  tokenizer_tgt: st.TokenizerWrapper, start_stop_tokens: list,
                  max_len, device):
    assert start_stop_tokens is not None and len(start_stop_tokens) == 2
    start_idx, end_idx = tokenizer_tgt.token_to_id(start_stop_tokens[0]), tokenizer_tgt.token_to_id(
        start_stop_tokens[1])
    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source_data_list, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(start_idx).type_as(source_data_list[0]).to(device)
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
            [decoder_input, torch.empty(1, 1).type_as(source_data_list[0]).fill_(next_token.item()).to(device)], dim=1
        )

        if next_token == end_idx:
            break

    return decoder_input.squeeze(0)


def beam_search_decode(model,
                       beam_size:int,
                       source_data_list:list[torch.Tensor],
                       source_mask,
                       tokenizer_tgt:st.TokenizerWrapper, start_stop_tokens: list,
                       max_len, device):
    assert start_stop_tokens is not None and len(start_stop_tokens) == 2
    start_idx, end_idx = tokenizer_tgt.token_to_id(start_stop_tokens[0]), tokenizer_tgt.token_to_id(
        start_stop_tokens[1])
    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source_data_list, source_mask)
    # Initialize the decoder input with the sos token
    decoder_initial_input = torch.empty(1, 1).fill_(start_idx).type_as(source_data_list[0]).to(device)

    # Create a candidate list
    candidates = [(decoder_initial_input, 1)]

    while True:

        # If a candidate has reached the maximum length, it means we have run the decoding for at least max_len iterations, so stop the search
        if any([cand.size(1) == max_len for cand, _ in candidates]):
            break

        # Create a new list of candidates
        new_candidates = []

        for candidate, score in candidates:

            # Do not expand candidates that have reached the EOS/END token
            if candidate[0][-1].item() == end_idx:
                continue

            # Build the candidate's mask
            candidate_mask = causal_mask(candidate.size(1)).type_as(source_mask).to(device)
            # calculate output
            out = model.decode(encoder_output, source_mask, candidate, candidate_mask)
            # get next token probabilities
            prob = model.project(out[:, -1])
            # get the top k candidates
            topk_prob, topk_idx = torch.topk(prob, beam_size, dim=1)
            for i in range(beam_size):
                # for each of the top k candidates, get the token and its probability
                token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)
                token_prob = topk_prob[0][i].item()
                # create a new candidate by appending the token to the current candidate
                new_candidate = torch.cat([candidate, token], dim=1)
                # We sum the log probabilities because the probabilities are in log space
                new_candidates.append((new_candidate, score + token_prob))

        # Sort the new candidates by their score
        candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
        # Keep only the top k candidates
        candidates = candidates[:beam_size]

        # If all the candidates have reached the EOS/END token, stop
        if all([cand[0][-1].item() == end_idx for cand, _ in candidates]):
            break

    # Return the best candidate
    return candidates[0][0].squeeze(0)


def run_validation(model, validation_ds: DataLoader,
                   tokenizer_tgt: st.TokenizerWrapper, start_stop_tokens: list,
                   max_len,
                   decode_algorithm,
                   device, print_msg, global_step, writer, num_examples=3,beam_size:int=3):
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
            encoder_input_list = batch["encoder_input"]  # [(b, I)]
            encoder_input_list = [t.to(device) for t in encoder_input_list]
            encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, I)

            # check that the batch size is 1
            assert encoder_input_list[0].size(0) == 1, "Batch size must be 1 for validation"

            if decode_algorithm == "beam":
                model_out = beam_search_decode(model, beam_size, encoder_input_list, encoder_mask,
                                               tokenizer_tgt,start_stop_tokens,
                                               max_len,device)
            else:
                assert decode_algorithm == "greedy"
                model_out = greedy_decode(model, encoder_input_list, encoder_mask, tokenizer_tgt, start_stop_tokens, max_len,
                                          device)
                
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
        # Compute the accuracy metric
        new_pred = [set(a.split(" ")) for a in predicted]
        new_expected = [set(a.split(" ")) for a in expected]
        acc = [len(a.intersection(b))/len(a) for a,b in zip(new_pred, new_expected)]
        acc = sum(acc) / len(acc)
        writer.add_scalar('validation accuracy', acc, global_step)
        writer.flush()

        # Compute the sklearn accuracy (?)
        avg_acc = 0
        for i, prex in enumerate(zip(predicted, expected)):
            pred = prex[0].split(" ")
            exp = prex[1].split(" ")
            exp = list(filter(lambda x: len(x) > 1, exp))
            pred = list(filter(lambda x: len(x) > 1, pred))
            if len(pred) < 10:
                pred.extend([''] * (10-len(pred)))
            if len(exp) < len(pred):
                exp.extend([''] * (len(pred)-len(exp)))
            avg_acc += accuracy_score(exp,pred)
        avg_acc /= len(expected)
        writer.add_scalar("sklearn acc", avg_acc, global_step)
        writer.flush()

        avg_f1 = 0
        for i, prex in enumerate(zip(predicted, expected)):
            pred = prex[0].split(" ")
            exp = prex[1].split(" ")
            exp = list(filter(lambda x: len(x) > 1, exp))
            pred = list(filter(lambda x: len(x) > 1, pred))
            if len(pred) < 10:
                pred.extend([''] * (10-len(pred)))
            if len(exp) < len(pred):
                exp.extend([''] * (len(pred)-len(exp)))
            avg_f1 += f1_score(exp, pred, average="micro")
        avg_f1 /= len(expected)
        writer.add_scalar("f1", avg_f1, global_step)
        writer.flush()

        avg_precision = 0
        for i, prex in enumerate(zip(predicted, expected)):
            pred = prex[0].split(" ")
            exp = prex[1].split(" ")
            exp = list(filter(lambda x: len(x) > 1, exp))
            pred = list(filter(lambda x: len(x) > 1, pred))
            if len(pred) < 10:
                pred.extend([''] * (10-len(pred)))
            if len(exp) < len(pred):
                exp.extend([''] * (len(pred)-len(exp)))
            avg_precision += precision_score(exp, pred, average="micro")
        avg_f1 /= len(expected)
        writer.add_scalar("precision", avg_precision, global_step)
        writer.flush()

        avg_recall = 0
        for i, prex in enumerate(zip(predicted, expected)):
            pred = prex[0].split(" ")
            exp = prex[1].split(" ")
            exp = list(filter(lambda x: len(x) > 1, exp))
            pred = list(filter(lambda x: len(x) > 1, pred))
            if len(pred) < 10:
                pred.extend([''] * (10-len(pred)))
            if len(exp) < len(pred):
                exp.extend([''] * (len(pred)-len(exp)))
            avg_recall += recall_score(exp, pred, average="micro")
        avg_f1 /= len(expected)
        writer.add_scalar("recall", avg_recall, global_step)
        writer.flush()

def get_tokenizers(config) -> ((st.ConcatTokenizer | list[st.TokenizerWrapper]), st.TokenizerWrapper):
    SPACE_SPLITTER = st.Splitter(lambda input_str: input_str.split(), config["list_elem_separation_token"])
    HEX_VOCAB = ["0x"] + [hex(j)[2:].zfill(2) for j in range(0xff + 1)]
    HEX_ADDRESS_SPLITTER = st.Splitter(lambda address: [address[i:i + 2] for i in range(0, len(address), 2)])
    SPECIAL_TOKENS = config["bpe_special_tokens"]
    def get_feature_tokenizer(tok_file_or_custom: str, feature: Feature, padder=False,
                              sentence_like_wrap_mode: str | None = None) -> st.TokenizerWrapper:
        # Since '!' is not an allowed character in path, we use it as a flag to signal if we should prefer using a custom
        # vocab tokenizer for the features that support it
        # TODO: if you want to refactor, passing along a bool flag in all the funcs is infinitely better, but couldn't be arsed to do it
        custom = "!" in tok_file_or_custom
        tok_file_or_custom = tok_file_or_custom.replace('!','')

        is_list = "list" in feature.type
        primitive_feature_type = feature.get_primitive_type()
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

        if custom and "hex" in primitive_feature_type or "bit" in primitive_feature_type:
            if "hex" in primitive_feature_type:
                custom_vocab = HEX_VOCAB
                if "number" not in primitive_feature_type:
                    input_splitter = HEX_ADDRESS_SPLITTER
                else:
                    # our input is going to be either 0x{2*N} or 0x{2*N+1}
                    # In the first case, no problem. In the second case however, we must add a zero after the 0x for our
                    # tokenizer to work
                    input_splitter = st.Splitter(
                        lambda number: [number[i:i + 2] for i in range(0, len(number), 2)] if len(number) % 2 == 0
                                  else [number[:2],'0'+number[2]] + [number[i:i+2] for i in range(3,len(number),2)]
                    )
            else:  # "bit" in prim_feature_type
                custom_vocab = ['0', '1']
                input_splitter = st.Splitter(lambda bitmap: [bitmap[i] for i in range(len(bitmap))])

            tokenizer = st.SimpleCustomVocabTokenizer(custom_vocab, SPECIAL_TOKENS, input_splitter=input_splitter)
        else:
            tokenizer_path = Path(tok_file_or_custom.format(primitive_feature_type))
            print(tokenizer_path)
            assert tokenizer_path.exists() and tokenizer_path.suffix == ".json"
            tokenizer = Tokenizer.from_file(tokenizer_path.absolute().as_posix())

        return st.TokenizerWrapper(
            tokenizer,
            len(SPECIAL_TOKENS),
            feature.max_len,
            splitter=splitter,
            pad_token=pad_token,
            wrap_parameters=wrap_params)

    def get_feature_tokenizer_dict(tok_file_or_custom, features: list[Feature], all_padders=False):
        ret_dict = {}
        for feature in features:
            sentence_like_wrap_mode = None
            if feature.name == "prev_faults":
                sentence_like_wrap_mode = "insert_lr"
            ret_dict[feature.name] = get_feature_tokenizer(tok_file_or_custom, feature, padder=all_padders,
                                                           sentence_like_wrap_mode=sentence_like_wrap_mode)
        return ret_dict

    def helper_get_tok_list_from_dict(features, feature_tokenizer_dict):
        return [feature_tokenizer_dict[feature.name] for feature in features]

    base_tokenizer = config["base_tokenizer"]
    emb_type = config["embedding_technique"]
    tok_file = config['tokenizer_files']
    input_features, output_features = config["input_features"], config["output_features"]
    assert len(output_features) == 1 and "list" in output_features[0].type
    # To build the tokenizers, see make_tokens.py
    if base_tokenizer == "bpe":
        out_tokenizer = get_feature_tokenizer(tok_file, output_features[0], padder=True,
                                          sentence_like_wrap_mode="no_insert_get_right_index")
    elif base_tokenizer == "hextet":
        out_tokenizer = st.TokenizerWrapper(
            st.SimpleCustomVocabTokenizer(HEX_VOCAB,SPECIAL_TOKENS,HEX_ADDRESS_SPLITTER),
            len(SPECIAL_TOKENS),
            output_features[0].max_len,
            splitter=SPACE_SPLITTER,
            pad_token=config["pad_token"],
            wrap_parameters=st.WrapParameters(
                config["start_stop_generating_tokens"],
                "no_insert_get_right_index")
        )
    else:
        assert base_tokenizer == "text"
        raise NotImplementedError
    #if token_type in ["concat_tokens", "hextet_concat"]:
    individual_padders = emb_type in ["meta_transformer","embed_concat"]
    # Since '!' is not an allowed character in path, we use it as a flag to signal if we should prefer using a custom
    # vocab tokenizer for the features that support it
    # TODO: if you want to refactor, passing along a bool flag in all the funcs is infinitely better, but couldn't be arsed to do it
    tok_file = ('!' if base_tokenizer == 'hextet' else '') + tok_file
    inp_ret_dict = get_feature_tokenizer_dict(tok_file, input_features, all_padders=individual_padders)
    if emb_type == "tok_concat":
        final_input_tokenizer = st.ConcatTokenizer(
                config["feature_separation_token"],
                config["pad_token"],
                helper_get_tok_list_from_dict(input_features, inp_ret_dict))
        return final_input_tokenizer, out_tokenizer
    elif emb_type in ["meta_transformer", "embed_concat"]:
        # Since each feature will be embedded differently, each tokenizer must be its own padder --> must recreate dict
        return helper_get_tok_list_from_dict(input_features, inp_ret_dict), out_tokenizer
    else:
        assert emb_type == "onetext"
        raise NotImplementedError
        tokenizer_path = Path(tok_file.format("onetext"))
        assert tokenizer_path.exists() and tokenizer_path.suffix == ".json"
        inp_ret_dict = Tokenizer.from_file(tokenizer_path.absolute().as_posix())
        # TODO: Use gpt model or custom build bpe, can be loaded online; c.f. tokenizers doc

def get_ds(config, generator) -> (
        DataLoader, DataLoader, (st.ConcatTokenizer | list[st.TokenizerWrapper]), st.TokenizerWrapper):
    valid_df = None
    # It only has the train split, so we divide it overselves
    if "processed" in config["data_path"]:
        df_raw = read_csv(config["data_path"])
    else:
        trace_type = config["trace_type"]
        if trace_type == "bpftrace":
            df_raw = process_bpftrace(config["data_path"], config["past_window"], config["k_predictions"],config["page_masked"],
                             save=None)  # load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
        else:
            assert trace_type == "fltrace"
            train_type = config["train_on_trace"]
            if train_type == "2train_1test":
                df_files_list = config["data_path"]
                assert type(df_files_list) == list and len(df_files_list) == 3
                df_raw = process_fltrace([df_files_list[0],df_files_list[-1]],config["objdump_path"],config["past_window"], config["k_predictions"],config["page_masked"],config["code_window"],multiple=True,save=None)
                valid_df = process_fltrace(df_files_list[1],config["objdump_path"],config["past_window"], config["k_predictions"],config["page_masked"],config["code_window"],multiple=False,save=None)
            else:
                df_raw = process_fltrace(config["data_path"], config["objdump_path"], config["past_window"],
                                         config["k_predictions"], config["page_masked"], config["code_window"],
                                         save=None)
    df_raw = df_raw.astype({col: str for col in df_raw.columns if df_raw[col].dtype == "int64"})
    if valid_df is not None:
        valid_df = valid_df.astype({col: str for col in valid_df.columns if valid_df[col].dtype == "int64"})


    print("loaded data")
    # Build tokenizers
    src_tokenizer, tgt_tokenizer = get_tokenizers(config)
    if valid_df is None:
        train_tensor_size = int(config["train_test_split"] * len(df_raw))
        indices = torch.arange(
            len(df_raw))  # torch.randperm(len(df_raw), generator=generator)  # TODO think about overlapping pfault windows

        subsample_rate = config["subsample"]
        train_ds = PageFaultDataset(config, df_raw, indices[:train_tensor_size],
                                    src_tokenizer,
                                    tgt_tokenizer,
                                    sample_percentage=subsample_rate
                                    )
        val_ds = PageFaultDataset(config, df_raw, indices[train_tensor_size:],
                                  src_tokenizer,
                                  tgt_tokenizer,
                                  sample_percentage=subsample_rate
                                  )
    else:
        subsample_rate = config["subsample"]
        indices_train = list(range(len(df_raw)))
        train_ds = PageFaultDataset(config, df_raw, indices_train,
                                    src_tokenizer,
                                    tgt_tokenizer,
                                    sample_percentage=subsample_rate
                                    )
        indices_test = list(range(len(valid_df)))
        val_ds = PageFaultDataset(config, df_raw, indices_test,
                                  src_tokenizer,
                                  tgt_tokenizer,
                                  sample_percentage=subsample_rate
                                  )

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer


def get_model(config, inp_tokenizer: st.ConcatTokenizer | list[st.TokenizerWrapper] | st.TokenizerWrapper,
              out_tokenizer: st.TokenizerWrapper):
    base_tokenizer = config["base_tokenizer"]
    emb_type = config["embedding_technique"]
    out_vocab_size = out_tokenizer.get_vocab_size()
    out_seq_len = int(config["output_features"][0].max_len)
    used_features = config["input_features"]
    if emb_type == "tok_concat":
        src_vocab_size = inp_tokenizer.get_vocab_size()
        inp_seq_len = (sum([int(feature.max_len) for feature in used_features]) + len(
            used_features) - 1)  # we add a separator token between features
    elif emb_type == "onetext":
        assert type(inp_tokenizer) == st.TokenizerWrapper
        src_vocab_size = inp_tokenizer.get_vocab_size()
        # inp_seq_len = ? TODO
        raise NotImplementedError
    else:
        assert emb_type in ["meta_transformer", "embed_concat"] and type(inp_tokenizer) == list
        src_vocab_size = [it.get_vocab_size() for it in inp_tokenizer]
        inp_seq_len = [int(feature.max_len) for feature in used_features]
        # TODO: we might have to slightly adapt build_model for those two, which is why I am currently taking the tokenizers themselves as input
        # TODO cont: This might however not be needed, maybe we can do with only the different sizes/lengths


    model = build_model(config, src_vocab_size, out_vocab_size, inp_seq_len, out_seq_len)
    return model


def train_model(config,mass_training=False):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print(
            "      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print(
            "      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Check if we're in the middle of a "train" : if yes, load the current random sample (based off the seed)
    # Else, take (new) random seed
    was_currently_training = len(source_model_files(config)) != 0
    generator = torch.Generator(device)
    mp = Path(get_model_full_path(config))
    # Make sure the model's run folder exists
    mp.mkdir(parents=True, exist_ok=True)
    # Also create the generator folder
    gen_dir = mp / GENERATOR_PREFIX
    gen_dir.mkdir(parents=True, exist_ok=True)
    if was_currently_training:
        seed_path = gen_dir / SEED_FN
        assert seed_path.exists()
        seed = torch.load(seed_path.absolute().as_posix())
        generator = generator.manual_seed(seed)
        state_path = gen_dir / STATE_FN
        assert state_path.exists()
        gen_state = torch.load(state_path.absolute().as_posix())
    else:
        torch.save(generator.seed(), (gen_dir / SEED_FN).as_posix())

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config, generator)
    # Save generator TODO: useless to do here, save at weight save time iff generator impacts training
    torch.save(generator.get_state(), (gen_dir / STATE_FN).as_posix())
    model = get_model(config, tokenizer_src, tokenizer_tgt).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_folder'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], eps=1e-6,amsgrad=True)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    past_save_files = []
    MAX_PREVIOUS_WEIGHTS_HISTORY = config["max_weight_save_history"]
    model_filename = latest_weights_file_path(config) if preload == 'latest' \
        else get_weights_file_path(config, preload) if preload \
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
        past_save_files = [Path(smf).as_posix() for smf in source_model_files(config)]
        past_save_files.sort()
    else:
        print(f'No model to preload, starting training of {get_model_full_path(config)} from scratch')

    emb_type = config["embedding_technique"]
    pad_token = config["pad_token"]
    if emb_type in ["meta_transformer", "embed_concat"]:
        assert type(tokenizer_src) == list
        input_pad_ids = [tok.token_to_id(pad_token) for tok in tokenizer_src]
        input_pad_id = input_pad_ids[0]
        for ipi in input_pad_ids:
            assert ipi == input_pad_id
    elif emb_type == "tok_concat":
        assert type(tokenizer_src) == st.ConcatTokenizer
        input_src_potential_pad_ids = tokenizer_src.token_to_id(pad_token)
        assert len(input_src_potential_pad_ids) == 1
        input_pad_id = input_src_potential_pad_ids[0]
    else:
        assert emb_type == "onetext"
        raise NotImplementedError

    output_pad_id = tokenizer_tgt.token_to_id(config["pad_token"])
    assert input_pad_id == output_pad_id

    ce_loss = nn.CrossEntropyLoss(ignore_index=output_pad_id,
                                  label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        if mass_training and len(past_save_files) > MAX_PREVIOUS_WEIGHTS_HISTORY:
            oldest_file = past_save_files.pop(0)
            Path(oldest_file).unlink()
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input_list = batch['encoder_input']  # [(B, I)]
            encoder_input_list = [t.to(device) for t in encoder_input_list]
            decoder_input = batch['decoder_input'].to(device)  # (B, O')
            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, I)
            decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, O', O')
            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input_list, encoder_mask)  # (B, I, D)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input,
                                          decoder_mask)  # (B, O', D)
            proj_output = model.project(decoder_output)  # (B, O', D)

            # Compare the output with the label
            label = batch['label'].to(device)  # (B, O')

            # Compute the loss using a simple cross entropy
            loss = ce_loss(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
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

        run_validation(model, val_dataloader, tokenizer_tgt, config["start_stop_generating_tokens"],
                       config["output_features"][0].max_len,config["decode_algorithm"],
                       device, lambda msg: batch_iterator.write(msg), global_step, writer,beam_size=config["beam_size"])

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        past_save_files.append(model_filename)

def multi_config_train():
    configs = get_all_configs()
    for c in configs:
        train_model(c)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    train_model(get_config(),True)
