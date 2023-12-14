import copy
from dataclasses import dataclass
import numpy as np

import tokenizers

from model import build_model
from dataset import PageFaultDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path, source_model_files, get_model_full_path, \
    SEED_FN, STATE_FN, GENERATOR_PREFIX

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
from data_parser import *
import warnings
from tqdm import tqdm
import os
from pathlib import Path
from pandas import read_csv

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

from trained_tokenizers import special_tokenizers as st

import torchmetrics
from torch.utils.tensorboard import SummaryWriter


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    print(torch.mean(encoder_output))
    print(torch.std(encoder_output))
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer,
                   num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

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


def get_feature_tokenizer(tok_file, feature, padder=False) -> st.TokenizerWrapper: #todo revert
    SPACE_SPLITTER = st.Splitter(lambda input_str: input_str.split(), config["list_elem_separation_token"])
    is_list = "list" in feature.type
    primitive_feature_type = feature.get_primitive_type()

    tokenizer_path = Path(tok_file.format(primitive_feature_type))
    assert tokenizer_path.exists() and tokenizer_path.suffix == ".json"
    tokenizer = Tokenizer.from_file(tokenizer_path.absolute().as_posix())
    splitter = None
    if is_list:
        splitter = SPACE_SPLITTER
    pad_token = config["pad_token"] if padder else None
    return st.TokenizerWrapper(
        tokenizer,
        len(config["bpe_special_tokens"]),
        feature.max_len,
        splitter,
        pad_token=pad_token)


def get_tokenizers(config) -> ((st.ConcatTokenizer | list[st.TokenizerWrapper]), st.TokenizerWrapper):
    SPACE_SPLITTER = st.Splitter(lambda input_str: input_str.split(), config["list_elem_separation_token"])

    def get_feature_tokenizer_dict(tok_file, features, all_padders=False):
        ret_dict = {}
        for feature in features:
            ret_dict[feature.type] = get_feature_tokenizer(tok_file, feature, padder=all_padders)
        return ret_dict

    def get_per_feature_tokenizers_list_from_dict(features, feature_tokenizer_dict):
        return [feature_tokenizer_dict[feature.type] for feature in features]

    token_type = config["embedding_technique"]
    tok_file = config['tokenizer_files']
    input_features, output_features = config["input_features"], config["output_features"]
    assert len(output_features) == 1
    # To build the tokenizers, see make_tokens.py
    out_tokenizer = get_feature_tokenizer(tok_file, output_features[0], padder=True)
    if token_type in ["concat_tokens", "hextet_concat"]:
        inp_ret_dict = get_feature_tokenizer_dict(tok_file, input_features)
        if token_type == "hextet_concat":
            for feature in input_features:
                prim_feature_type = feature.get_primitive_type()
                is_list = "list" in feature.type
                SPECIAL_TOKENS = config["bpe_special_tokens"]
                h_vocab = [hex(j)[2:].zfill(2) for j in range(int("0xff", 16))] + ["0x"]
                b_vocab = ['0', '1']
                splitter = None
                if is_list:
                    splitter = SPACE_SPLITTER
                if "hex" in prim_feature_type:
                    inp_ret_dict[prim_feature_type] = st.TokenizerWrapper(
                        st.SimpleCustomVocabTokenizer(h_vocab, SPECIAL_TOKENS),
                        len(SPECIAL_TOKENS),
                        feature.max_len,
                        splitter)
                elif "bit" in prim_feature_type:
                    inp_ret_dict[prim_feature_type] = st.TokenizerWrapper(
                        st.SimpleCustomVocabTokenizer(b_vocab, SPECIAL_TOKENS),
                        len(SPECIAL_TOKENS),
                        feature.max_len,
                        splitter)
        final_input_tokenizer = st.ConcatTokenizer(
            config["feature_separation_token"],
            config["pad_token"],
            get_per_feature_tokenizers_list_from_dict(input_features, inp_ret_dict))
        return final_input_tokenizer, out_tokenizer
    elif token_type in ["meta_transformer", "embed_concat"]:
        inp_ret_dict = get_feature_tokenizer_dict(tok_file, input_features, all_padders=True)
        # Since each feature will be embedded differently, each tokenizer must be its own padder --> must recreate dict
        return get_per_feature_tokenizers_list_from_dict(input_features, inp_ret_dict), out_tokenizer
    else:
        assert token_type == "onetext"
        raise NotImplementedError
        tokenizer_path = Path(tok_file.format("onetext"))
        assert tokenizer_path.exists() and tokenizer_path.suffix == ".json"
        inp_ret_dict = Tokenizer.from_file(tokenizer_path.absolute().as_posix())
        # TODO: Use gpt model or custom build bpe, can be loaded online; c.f. tokenizers doc


def get_ds(config, generator):
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

    model = build_model(config, src_vocab_size, out_vocab_size, inp_seq_len, out_seq_len)
    return model


def train_model(config):
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

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
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
            encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
            print(torch.mean(encoder_output))
            print(torch.std(encoder_output))
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input,
                                          decoder_mask)  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device)  # (B, seq_len)

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
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
                       lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    # warnings.filterwarnings("ignore")
    # train_model(get_config())
    config = get_config()
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

    feature_tokenizer = get_feature_tokenizer(config['tokenizer_files'], config['input_features'][3])
    dataframe = df_raw['ustack']
    lengths = []
    occurences = {}
    for data in dataframe:
        data_list = data #.values.flatten().tolist()[0]
        encoded = feature_tokenizer.encode(data_list)
        ids = encoded.ids
        lengths.append(len(ids))

        for id in ids:
            token = feature_tokenizer.id_to_token(id)
            occurences[token] = occurences.get(token, 0) + 1



    print(f"Mean length tokenized string: {np.mean(lengths)}")
    print(f"Std length tokenized string: {np.std(lengths)}")
    most_common_tokens = sorted(occurences, key=occurences.get, reverse=True)[:10]
    total_occurences = np.sum(list(occurences.values()))
    print("Most common tokens:")
    for token in most_common_tokens:
        print(f"\tToken {token} ({feature_tokenizer.token_to_id(token)}), total occurences = {occurences[token]}, percentage = {(occurences[token] / total_occurences)*100}")

    #create graph of occurences with tokens at the basis axis
    import matplotlib.pyplot as plt
    plt.bar(range(len(occurences)), list(occurences.values()), align='center')
    plt.xticks(range(len(occurences)), list(occurences.keys()))
    plt.show()

    plt.bar(range(len(most_common_tokens)), [occurences[token] for token in most_common_tokens], align='center')
    plt.xticks(range(len(most_common_tokens)), most_common_tokens)
    plt.show()

    token_ids = [feature_tokenizer.token_to_id(token) for token in most_common_tokens]
    #and one for most_common_tokens showing the token id on the basis axis
    plt.bar(range(len(most_common_tokens)), [occurences[token] for token in most_common_tokens], align='center')
    plt.xticks(range(len(most_common_tokens)), token_ids)
    plt.show()


