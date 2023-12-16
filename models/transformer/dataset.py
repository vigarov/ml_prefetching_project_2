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
        self.input_view = df[[feature.name for feature in self.input_features]].iloc[indices_split]
        self.output_view = df[[feature.name for feature in self.output_features]].iloc[indices_split]
        self.len = len(indices_split)
        self.tokenization_type = config["embedding_technique"]
        self.pad_token = config["pad_token"]
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        input_possible_pad_ids = input_tokenizer.token_to_id(self.pad_token)
        assert len(input_possible_pad_ids) == 1
        self.pad_token_ids = (
        torch.tensor([input_possible_pad_ids[0]]), torch.tensor([output_tokenizer.token_to_id(self.pad_token)]))
        self.start_stop_generating_tokens = config["start_stop_generating_tokens"]

    def __len__(self):
        return self.len

    def __getitem__(self, index) -> dict:
        if torch.is_tensor(index):
            index = index.toList()
        raw_input, raw_output = self.input_view.iloc[index].values.flatten().tolist(), self.output_view.iloc[
            index].values.flatten().tolist()
        assert len(raw_output) == 1
        raw_output = raw_output[0]
        assert len(raw_input) == len(self.input_features)
        assert type(raw_input) == list and type(raw_input[0]) == str
        assert type(raw_output) == str

        # Output tokenization is the same for everyone :D
        assert self.output_tokenizer.wraps() and self.output_tokenizer.returns_right_wrap_index()
        tokenized_output, right_index = self.output_tokenizer.encode(raw_output)
        assert tokenized_output.ids[right_index] == self.output_tokenizer.token_to_id(self.pad_token)
        if right_index > 0:
            assert tokenized_output.ids[right_index-1] != self.output_tokenizer.token_to_id(self.pad_token)
        tokenized_output = tokenized_output.ids
        tokenized_output = torch.tensor(tokenized_output, dtype=torch.int64)  # size = (O)


        # Tokenize input, and return accordingly
        if self.tokenization_type in ["concat_tokens", "hextet_concat", "onetext"]:
            # Same "type" of tokenizations, the difference lies in the tokenizer itself
            tok_input = raw_input
            if self.tokenization_type == "onetext":
                assert type(self.input_tokenizer) == st.TokenizerWrapper
                # Simply transform to a big string, whitespace?-separeted before giving it to the tokenizer
                tok_input = ' '.join(raw_input)
            else:
                assert type(self.input_tokenizer) == st.ConcatTokenizer
            tokenized_input = self.input_tokenizer.encode(raw_input).ids
            tokenized_input = torch.tensor(tokenized_input, dtype=torch.int64)  # size = (I)
        elif self.tokenization_type in ["meta_transformer", "embed_concat"]:
            assert type(self.input_tokenizer) == list[st.TokenizerWrapper]
            assert len(self.input_tokenizer) == len(raw_input)
            tokenized_input = [tok_i.encode(inp_i) for inp_i, tok_i in zip(raw_input, self.input_tokenizer)]



        # For decoding, we would like our input to be prepended by a "start generating" token, so our model learns how
        # how to generate the first token as well
        # Our ground truth however, mustn't have that token. It will however have a "stop generating" token, so our
        # model learns when to stop generating. Note that this last token might seem useless in our case, since we have
        # word address separators, however, it teaches the model to generate only a few predictions. This will be useful
        # later, if we want to make this number a variable.
        decoder_input = torch.concat([
            torch.tensor([self.output_tokenizer.token_to_id(self.start_stop_generating_tokens[0])]),
            tokenized_output
        ])  # size = O+1 = O'

        ground_truth = torch.concat([
            tokenized_output[:right_index], # Actual tokens
            torch.tensor([self.output_tokenizer.token_to_id(self.start_stop_generating_tokens[1])]),
            tokenized_output[right_index:-1] # Paddings
        ])  # size = O + 1 = O'

        # The encoder mask simply corresponds to all the tokenized input, that is not a padding token
        encoder_mask = tokenized_input != self.pad_token_ids[0]  # size = (I)
        # Resize for per Batch and per model step (c.f.: the decoder mask ; the encoder input will always stay the
        # same while the decoder is progressively allowed through) # TODO not entirely sure of the second "per" here
        encoder_mask = encoder_mask.unsqueeze(0).unsqueeze(0).int()  # size (1,1,I)
        # The decode mask is all tokens which are not a padding, while not allowing the decoder to "look forward"
        # (--> progressively allow more of the input)
        decoder_mask = (tokenized_output != self.pad_token_ids[1]).unsqueeze(0).int()  # size = (1,O')
        progressive_mask = causal_mask(tokenized_output.size(0))  # size = (1, O', O')
        decoder_mask = decoder_mask & progressive_mask  # size = (1, O') & (1, O', O') TODO = (1, O', O')  ?

        # For logging purpose, we also return the actual data, as text
        input_as_str = '\n'.join(raw_input)

        return {
            "encoder_input": tokenized_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": ground_truth,
            "src_text": input_as_str,
            "tgt_text": raw_output
        }


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
