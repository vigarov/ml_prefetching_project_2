import os
from pathlib import Path

import torch
import torchmetrics

from models.transformer.config import get_device, get_config, latest_weights_file_path
from models.transformer.dataset import causal_mask, PageFaultDataset
from tokenizers import Tokenizer

from models.transformer.train import get_model, get_tokenizers, greedy_decode, beam_search_decode
from models.transformer.trained_tokenizers.special_tokenizers import SimpleTokenIdList


class Inferer:
    """
    Class used to infer the output of a model
    """
    def __init__(self, config, model=None, device=None):
        """
        :param config: the configuration (see config.py)
        :param model: the model to use for inference (if None, will use the configuration)
        :param device: the device to use for inference (if None, will use the configuration)
        """
        src_tokenizer, tgt_tokenizer = get_tokenizers(config)
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.config = config
        self.device = get_device() if device is None else device
        path_weights = latest_weights_file_path(config)
        self.model = self.load_model(path_weights) if model is None else model
        self.decode_algorithm = config["decode_algorithm"]
        self.beam_size = config["beam_size"] #FIXME not implemented yet

    def infer(self, data, max_len, output_file=None):
        """
        Infer the output of the model for the given data and print the result to stdout (and to a file if specified)
        :param data: the data to infer from
        :param max_len: the maximum length of the output
        :param output_file: the file to write the output to (if None, will only print to stdout)
        :return: None
        """
        self.model.eval()
        source_texts = []
        inferred = []

        try:
            # get the console window width
            with os.popen('stty size', 'r') as console:
                _, console_width = console.read().split()
                console_width = int(console_width)
        except:
            # If we can't get the console width, use 80 as default
            console_width = 80

        with torch.no_grad():
            config = get_config()
            subsample_rate = config["subsample"]
            indices_train = list(range(len(data)))
            dataset = PageFaultDataset(config, data, indices_train,
                                       self.tokenizer_src,
                                       self.tokenizer_tgt,
                                       sample_percentage=subsample_rate
                                       )

            # assert(dataset.__len__() == 1)

            dp = dataset.__getitem__(0)
            encoder_input_list = dp["encoder_input"]  # [(b, I)]
            encoder_input_list = [t.to(self.device) for t in encoder_input_list]
            encoder_mask = dp["encoder_mask"].to(self.device)  # (b, 1, 1, I)

            # check that the batch size is 1
            assert encoder_input_list[0].size(0) == 1, "Batch size must be 1 for validation"

            if self.decode_algorithm == "beam":
                model_out = beam_search_decode(self.model, self.beam_size, encoder_input_list, encoder_mask,
                                               self.tgt_tokenizer, config["start_stop_generating_tokens"],
                                               max_len, self.device)
            else:
                assert self.decode_algorithm == "greedy"
                model_out = greedy_decode(self.model, encoder_input_list, encoder_mask, self.tgt_tokenizer, config["start_stop_generating_tokens"],
                                          max_len,
                                          self.device)

            source_text = dp["src_text"][0]
            target_text = dp["tgt_text"][0]
            model_out_text = self.tgt_tokenizer.decode(SimpleTokenIdList(model_out.detach().cpu().numpy().tolist()))

            source_texts.append(source_text)
            inferred.append(model_out_text)

            # Print the source, target and model output
            source_string = f"{f'SOURCE: ':>12}{source_text}"
            infer_string = f"{f'INFERRED: ':>12}{model_out_text}"
            print('-' * console_width)
            print(source_string)
            print(infer_string)
            print('-' * console_width)

            if output_file is not None:
                with open(output_file, "w") as f:
                    f.write(f"{source_string}\n")
                    f.write(f"{infer_string}\n")
            return model_out_text

    def load_model(self, path):
        self.model = get_model(self.config, self.src_tokenizer,
                               self.tgt_tokenizer).to(self.device)
        self.model.load_state_dict(torch.load(path)['model_state_dict'])
        return self.model


if __name__ == '__main__':
    inferer = Inferer(get_config())
    inferer.infer("0x55bb3580f018", 15)