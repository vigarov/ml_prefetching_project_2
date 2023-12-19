import numpy as np
import torch
from torch.utils.data import Dataset
from trained_tokenizers import special_tokenizers as st


class PageFaultDataset(Dataset):
    def __init__(self, config, df, indices_split,
                 input_tokenizer: st.ConcatTokenizer | list[st.TokenizerWrapper] | st.TokenizerWrapper,
                 output_tokenizer: st.TokenizerWrapper,
                 sample_percentage:float=1.00  # lower than 1 if you want to subsample
                 ):
        self.input_features = config["input_features"]
        self.output_features = config["output_features"]
        self.input_view = df[[feature.name for feature in self.input_features]].iloc[indices_split]
        self.output_view = df[[feature.name for feature in self.output_features]].iloc[indices_split]
        self.subsample_skip = (1/sample_percentage)-1
        self.len = int(len(indices_split) * sample_percentage)
        self.base_tokenizer = config["base_tokenizer"]
        self.embedding_type = config["embedding_technique"]
        self.pad_token = config["pad_token"]
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        if self.embedding_type in ["embed_concat","meta_transformer"]:
            assert type(input_tokenizer) == list and len(input_tokenizer) > 0
            inp_pad_id = input_tokenizer[0].token_to_id(self.pad_token)
            # For simplicity, assume they all have the same pad id
            for it in input_tokenizer:
                assert it.token_to_id(self.pad_token) == inp_pad_id
        elif self.embedding_type == "tok_concat":
            assert type(input_tokenizer) == st.ConcatTokenizer
            input_possible_pad_ids = input_tokenizer.token_to_id(self.pad_token)
            assert len(input_possible_pad_ids) == 1
            inp_pad_id = input_possible_pad_ids[0]
        else:
            assert self.embedding_type == "onetext"
            inp_pad_id = input_tokenizer.token_to_id(self.pad_token)
        self.pad_token_ids = (torch.tensor([inp_pad_id]), torch.tensor([output_tokenizer.token_to_id(self.pad_token)]))
        self.start_stop_generating_tokens = config["start_stop_generating_tokens"]

    def __len__(self):
        return self.len

    def __getitem__(self, index) -> dict:
        if torch.is_tensor(index):
            index = index.toList()
        index = [i+self.subsample_skip for i in index]
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

        # The output/decoder-input tokenization is the same accross our configs, we do it here first
        tokenized_output = tokenized_output.ids
        tokenized_output = torch.tensor(tokenized_output, dtype=torch.int64)  # size = (O)

        # For decoding, we would like our input to be prepended by a "start generating" token, so our model learns how
        # how to generate the first token as well
        # Our ground truth however, mustn't have that token. It will however have a "stop generating" token, so our
        # model learns when to stop generating. Note that this last token might seem useless in our case, since we have
        # word address separators, however, it teaches the model to generate only a few predictions. This will be useful
        # later, if we want to make this number a variable.
        decoder_input = torch.concat([
            torch.tensor([self.output_tokenizer.token_to_id(self.start_stop_generating_tokens[0])]),
            tokenized_output[:-1]
        ])  # size = O+1 = O'

        ground_truth = torch.concat([
            tokenized_output[:right_index], # Actual tokens
            torch.tensor([self.output_tokenizer.token_to_id(self.start_stop_generating_tokens[1])]),
            tokenized_output[right_index:-1] # Paddings
        ])  # size = O + 1 = O'

        assert decoder_input.size() == ground_truth.size()

        # The decode mask is all tokens which are not a padding, while not allowing the decoder to "look forward"
        # (--> progressively allow more of the input)
        decoder_mask = (decoder_input != self.pad_token_ids[1]).unsqueeze(0).int()  # size = (1,O')
        progressive_mask = causal_mask(decoder_input.size(0))  # size = (1, O', O')
        decoder_mask = decoder_mask & progressive_mask  # size = (1, O') & (1, O', O') TODO = (1, O', O')  ?


        # Tokenize input (encoding), and return accordingly
        if self.embedding_type in ["tok_concat", "onetext"]:
            # Same "type" of tokenization output, the difference only lies in the tokenizer itself
            tok_input = raw_input
            if self.embedding_type == "onetext":
                assert type(self.input_tokenizer) == st.TokenizerWrapper
                # Simply transform to a big string, whitespace?-separeted before giving it to the tokenizer
                tok_input = ' '.join(raw_input)
            else:
                assert type(self.input_tokenizer) == st.ConcatTokenizer

            tokenized_input = self.input_tokenizer.encode(tok_input).ids
            tokenized_input = torch.tensor(tokenized_input, dtype=torch.int64)  # size = (I)
            # The encoder mask simply corresponds to all the tokenized input, that is not a padding token
            encoder_mask = tokenized_input != self.pad_token_ids[0]  # size = (I)
            tokenized_input = [tokenized_input]  # size = [(I)]
        else:
            assert self.embedding_type in ["meta_transformer", "embed_concat"]
            assert type(self.input_tokenizer) == list  # [st.TokenizerWrapper]
            assert len(self.input_tokenizer) == len(raw_input)
            tokenized_input_list: list[torch.Tensor] = [torch.tensor(tok_i.encode(inp_i).ids, dtype=torch.int64) for inp_i, tok_i in zip(raw_input, self.input_tokenizer)]
            # Important Note: the encoder_mask is applied *after* the concatenation of the embeddings (if embed_concat/meta_transformer)
            # --> we don't need to add yet another artificial dimension
            # Note2: PyCharm's warning below is straight up wrong - we are definitely inputting a list of Tensors
            encoder_mask = torch.hstack([tok_i != self.pad_token_ids[0] for tok_i in tokenized_input_list])
            tokenized_input = tokenized_input_list

        # Resize for per Batch and per model step (c.f.: the decoder mask ; the encoder input will always stay the
        # same while the decoder is progressively allowed through) # TODO not entirely sure of the second "per" here
        encoder_mask = encoder_mask.unsqueeze(0).unsqueeze(0).int()  # size (1,1,I)

        # For logging purpose, we also return the actual data, as text
        input_as_str = '\n'.join(raw_input)
        return {                               # Sizes     Empty = Unchanged from Left
                                               #                K - features
                                               # Concat      //  Embed-Meta  // OneText
            "encoder_input": tokenized_input,  # (I)              [(I')]
            "encoder_mask": encoder_mask,      # (1,1,I)          (1,1,I)
            "decoder_input": decoder_input,    # O'
            "decoder_mask": decoder_mask,      # (1,O',O')
            "label": ground_truth,             # O'
            "src_text": input_as_str,
            "tgt_text": raw_output
        }



def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
