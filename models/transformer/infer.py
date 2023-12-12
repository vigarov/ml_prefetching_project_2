import os
from pathlib import Path

import torch
import torchmetrics

from models.transformer.config import get_device, get_config
from models.transformer.dataset import causal_mask, BilingualDataset
from tokenizers import Tokenizer

from models.transformer.train import get_model


class Inferer:
    def __init__(self, config, model=None, device=None):
        tokenizer_src_path = Path(config['tokenizer_file'].format(config['lang_src']))
        tokenizer_tgt_path = Path(config['tokenizer_file'].format(config['lang_tgt']))

        tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
        tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = config['lang_src']
        self.tgt_lang = config['lang_tgt']
        self.config = config
        self.device = get_device() if device is None else device
        self.model = self.load_model("canneal_trace_v1_weights/tmodel_03.pt") if model is None else model #todo modify for portability

    def greedy_decode(self, model, source, source_mask, tokenizer_tgt, max_len, device):
        #todo refactor to use class variables
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


    def infer(self, data, max_len, output_file=None):
        self.model.eval()
        source_texts = []
        predicted = []
        dataAdapted = {}
        dataAdapted[0] = {self.src_lang: data, self.tgt_lang: ""}

        try:
            # get the console window width
            with os.popen('stty size', 'r') as console:
                _, console_width = console.read().split()
                console_width = int(console_width)
        except:
            # If we can't get the console width, use 80 as default
            console_width = 80

        with torch.no_grad():

            dataset = BilingualDataset(dataAdapted, self.tokenizer_src, self.tokenizer_tgt, self.src_lang, self.tgt_lang,
                                       max_len)

            #assert(dataset.__len__() == 1)

            dp = dataset.__getitem__(0)
            encoder_input = dp["encoder_input"].to(self.device)  # (b, seq_len)
            encoder_mask = dp["encoder_mask"].to(self.device)  # (b, 1, 1, seq_len)


            model_out = self.greedy_decode(self.model, encoder_input, encoder_mask, self.tokenizer_tgt, max_len, self.device)

            source_text = dp["src_text"][0]
            # target_text = dp["tgt_text"][0]
            model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            # expected.append(target_text)
            predicted.append(model_out_text)

            model_out = self.greedy_decode(self.model, encoder_input, encoder_mask, self.tokenizer_tgt, max_len, self.device)

            model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(data)
            predicted.append(model_out_text)

            # Print the source, target and model output
            print('-' * console_width)
            print(f"{f'SOURCE: ':>12}{data}")
            print(f"{f'PREDICT:  ':>12}{model_out_text}")

            if output_file is not None:
                output_file.write(f"{model_out_text}\n")

    def load_model(self, path):
        self.model = get_model(self.config, self.tokenizer_src.get_vocab_size(), self.tokenizer_tgt.get_vocab_size()).to(
            self.device)
        self.model.load_state_dict(torch.load(path)['model_state_dict'])
        return self.model




if __name__ == '__main__':
    inferer = Inferer(get_config())
    inferer.infer("0x55bb3580f018", 15)
