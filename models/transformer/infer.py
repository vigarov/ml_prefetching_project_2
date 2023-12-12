import os
from pathlib import Path

import torch
import torchmetrics

from models.transformer.config import get_device
from models.transformer.dataset import causal_mask
from tokenizers import Tokenizer

class Inferer:
    def __init__(self, config, model):
        tokenizer_src_path = Path(config['tokenizer_src_file'])
        tokenizer_tgt_path = Path(config['tokenizer_tgt_file'])
        tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
        tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

        self.src_lang = config['lang_src']
        self.tgt_lang = config['lang_tgt']

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def greedy_decode(self, model, source, source_mask, tokenizer_tgt, max_len, device):
        sos_idx = tokenizer_tgt.token_to_id('[SOS]')
        eos_idx = tokenizer_tgt.token_to_id('[EOS]')

        # Precompute the encoder output and reuse it for every step
        encoder_output = model.encode(source, source_mask)
        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
        while True:
            if decoder_input.size(1) == max_len:
                break

            # build mask for target
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(
                device)  # TODO check if we have to adapt for other datasets

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


    def infer(self, model, data, max_len, device=None, output_file=None):
        if device is None:
            device = get_device()

        model.eval()
        source_texts = []
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

            # Transform the text into tokens
            enc_input_tokens = self.tokenizer_src.encode(data).ids

            # Add sos, eos and padding to each sentence
            enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
            #todo do we pass only one sentence in inference? or multiple? bc if one sentence computation is not correct

            # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
            if enc_num_padding_tokens < 0:
                raise ValueError("Sentence is too long")

            # Add <s> and </s> token
            encoder_input = torch.cat(
                [
                    self.sos_token,
                    torch.tensor(enc_input_tokens, dtype=torch.int64),
                    self.eos_token,
                    torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
                ],
                dim=0,
            )

            encoder_input = encoder_input # (b, seq_len)
            encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()  # (b, 1, 1, seq_len)

            model_out = self.greedy_decode(model, encoder_input, encoder_mask, max_len, device)

            model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(data)
            predicted.append(model_out_text)

            # Print the source, target and model output
            print('-' * console_width)
            print(f"{f'SOURCE: ':>12}{data}")
            print(f"{f'PREDICT:  ':>12}{model_out_text}")

            if output_file is not None:
                output_file.write(f"{model_out_text}\n")


def load_model(path, model=None):
    if model is None:
        model = torch.load(path)
    else:
        model.load_state_dict(torch.load(path))
    return model


if __name__ == '__main__':
    model = load_model("models/transformer/transformer.pt")
