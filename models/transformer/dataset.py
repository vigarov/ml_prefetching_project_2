import numpy as np
import torch
from torch.utils.data import Dataset
from trained_tokenizers import special_tokenizers as st


class PageFaultDataset(Dataset):
    def __init__(self, config, df, indices_split,
                 input_tokenizer: st.ConcatTokenizer | list[st.TokenizerWrapper] | st.TokenizerWrapper,
                 output_tokenizer: st.TokenizerWrapper):
        self.input_features = config["input_features"]
        self.output_features = config["output_features"]
        self.ind_split = indices_split # TODO: remove
        self.input_view = df[[feature.name for feature in self.input_features]].iloc[indices_split]
        self.output_view = df[[feature.name for feature in self.output_features]].iloc[indices_split]
        self.len = len(indices_split)
        self.tokenization_type = config["embedding_technique"]
        self.pad_token = config["pad_token"]
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        input_possible_pad_ids = input_tokenizer.token_to_id(self.pad_token)
        assert len(input_possible_pad_ids) == 1
        self.pad_token_ids = (torch.tensor([input_possible_pad_ids[0]]),torch.tensor([output_tokenizer.token_to_id(self.pad_token)]))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.toList()
        raw_input, raw_output = self.input_view.iloc[index].values.flatten().tolist(), self.output_view.iloc[index].values.flatten().tolist()
        assert len(raw_output) == 1
        raw_output = raw_output[0]
        assert len(raw_input) == len(self.input_features)
        assert type(raw_input) == list and type(raw_input[0]) == str
        assert type(raw_output) == str

        if self.tokenization_type != "onetext":
            tokenized_output = self.output_tokenizer.encode(raw_output).ids

        # Tokenize
        if self.tokenization_type in ["concat_tokens", "hextet_concat"]:
            tokenized_input = self.input_tokenizer.encode(raw_input).ids
        elif self.tokenization_type == "meta_transformer":
            raise NotImplementedError
        elif self.tokenization_type == "embed_concat":
            raise NotImplementedError
        else:
            assert self.tokenization_type == "onetext"
            # Simply transform to a big string, whitespace?-separeted before giving it to the tokenizer
            tokenized_input = self.input_tokenizer.encode(' '.join(raw_input)).ids
            tokenized_output = self.output_tokenizer.encode(' '.join(raw_output)).ids

        # Denote by L the concatenated sequence length (L might differ depending on tokenization type

        tokenized_input = torch.tensor(tokenized_input, dtype=torch.int64)  # size = (L)
        tokenized_output = torch.tensor(tokenized_output, dtype=torch.int64)  # size = (L)
        # The encoder mask simply corresponds to all the tokenized input, that is not a padding token
        encoder_mask = tokenized_input != self.pad_token_ids[0]  # size = (L)
        # Resize for per Batch and per model step (c.f.: the decoder mask ; the encoder input will always stay the
        # same while the decoder is progressively allowed through) # TODO not entirely sure of the second "per" here
        encoder_mask = encoder_mask.unsqueeze(0).unsqueeze(0).int()  # size (1,1,L)
        # The decode mask is all tokens which are not a padding, while not allowing the decoder to "look forward"
        # (--> progressively allow more of the input)
        decoder_mask = (tokenized_output != self.pad_token_ids[1]).unsqueeze(0).int()
        progressive_mask = causal_mask(tokenized_output.size(0))
        decoder_mask = decoder_mask & progressive_mask  # size = (1, L) & (1, L, L) TODO = (1, L, L)  ?

        # In our case, the ground truth is simply the fully decoded output
        ground_truth = tokenized_output.detach().clone()  # size = (L)

        # For logging purpose, we also return the actual data, as text
        input_as_str = '||'.join(raw_input)

        return {
            "encoder_input": tokenized_input,
            "decoder_input": tokenized_output,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": ground_truth,
            "src_text": input_as_str,
            "tgt_text": raw_output
        }


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
