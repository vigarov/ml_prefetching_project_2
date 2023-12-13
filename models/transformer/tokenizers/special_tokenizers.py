import types
from dataclasses import dataclass, field
from tokenizers import Tokenizer
import numpy as np

DEBUG_TOKENIZERS = True


def list_split(sequence, sep):
    chunk = []
    for val in sequence:
        if val == sep:
            yield chunk
            chunk = []
        else:
            chunk.append(val)
    yield chunk


@dataclass
class SimpleTokenIdList:
    ids: list[int] = field(default_factory=list[int])


@dataclass
class TokensList:
    tokens: list[str] = field(default_factory=list[str])


class SimpleCustomVocabTokenizer:
    def __init__(self, vocab, special_tokens):
        self.reverse_dict = {tok: i for i, tok in enumerate(special_tokens + vocab)}

    def tokenize(self, *inp):
        all_tokens = SimpleTokenIdList([])
        for address in inp:
            all_tokens.ids += self.__parse_address(address)

    def __parse_address(self, addr):
        addr = addr.strip()
        assert "0x" == addr[:2]
        return [self.reverse_dict[octet] if octet in self.reverse_dict.keys() else self.reverse_dict["[UNK]"] for octet
                in addr[::2]]

    def get_vocab_size(self):
        return len(self.reverse_dict.keys())


class ConcatTokenizer:
    def __init__(self, feature_separator_token, *tokenizers):
        self.feature_separator_token = feature_separator_token
        self.tokenizers = list(tokenizers)
        vocab_cumsums = np.cumsum(
            [tokenizer.get_vocab_size() for tokenizer in self.tokenizers]) + 1  # +1 to account for the fsp token
        self.vocab_size = vocab_cumsums[-1]
        vocab_cumsums = vocab_cumsums.tolist()
        vocab_cumsums.insert(0, 0)  # la tete a Toto
        self.vocab_cumsums = vocab_cumsums

    def get_vocab_size(self):
        return self.vocab_size

    def token_to_id(self, token_str):
        # Returns a list of all possible ids, as a token might be present in > 1 (sub) tokenizer
        if token_str == self.feature_separator_token:
            return 0
        all_possible_ids = []
        for tokenizer in self.tokenizers:
            ret = tokenizer.token_to_id(token_str)
            if ret and ret != "":
                all_possible_ids.append(ret)
        return all_possible_ids

    def id_to_token(self, id_):
        assert id_ >= 0
        if id_ == 0:
            return self.feature_separator_token
        else:
            for i, s in self.vocab_cumsums:
                if id_ < s:
                    return self.tokenizers[i - 1].id_to_token()
            raise LookupError

    def encode(self, features_to_encode: list) -> SimpleTokenIdList:
        assert len(features_to_encode) == len(self.tokenizers)
        ret = SimpleTokenIdList()
        for i, feature_value in enumerate(features_to_encode):
            encoded_tokens_list = (np.array(self.tokenizers[i].encode(feature_value)) + self.vocab_cumsums[i]).tolist()
            ret.ids += encoded_tokens_list
            if i != len(features_to_encode) - 1:
                ret.ids += [0]  # Add feature separator token
        return ret

    def decode(self, ids_list: SimpleTokenIdList) -> TokensList:
        # Will never be used in our model, as for the output (which is the only place we'll need `decode()`), we only
        # have one feature --> will not be wrapped around ConcatTokenizer
        if not DEBUG_TOKENIZERS:
            raise PermissionError

        # An implementation of the method can still be found down below though:
        per_tokenizer_lists = list(list_split(ids_list.ids, 0))
        assert len(per_tokenizer_lists) == len(self.tokenizers)
        ret = TokensList()
        for i, tokenizer_list in enumerate(per_tokenizer_lists):
            tokenizer_list = (np.array(tokenizer_list) - self.vocab_cumsums[i]).to_list()
            out_str = self.tokenizers[i].decode(tokenizer_list)
            ret.tokens.append(out_str)
        return ret


@dataclass
class Splitter:
    split_function: types.FunctionType  # function taking as argument str, returning list(str)
    replace_token: str


class BPEWrapper:
    # Important note: this wrapper only preserves the tokenizer's `decode(encode(x)) == x` propery iff the given
    # Splitter's replace_token corresponds to, when added, to the "reverse" of the split_function
    # (e.g.: if split_function == split() and replace_token = " ")
    def __init__(self, bpe_tokenizer, num_special_tokens: int, splitter: Splitter = None):
        self.bpe_tokenizer = bpe_tokenizer
        self.splitter = splitter
        self.vocab_size = bpe_tokenizer.get_vocab_size()
        # if we end up having a splitter replacement token, we want its id to be the last of the special tokens ones
        # (so that pad and unk are always at the same pos, might be important)
        self.num_special_tokens = num_special_tokens
        self.has_replacement_token = self.splitter is not None and self.splitter.replace_token != ""
        if self.has_replacement_token:
            assert self.splitter.replace_token not in self.bpe_tokenizer.get_vocab()
            self.vocab_size += 1

    def get_vocab_size(self):
        return self.vocab_size

    def token_to_id(self, token_str: str):
        if self.has_replacement_token:
            if token_str == self.splitter.replace_token:
                return self.num_special_tokens
            token_id = self.bpe_tokenizer.token_to_id(token_str)
            if token_id is None:
                return token_id
            assert isinstance(token_id, int)
            if token_id < self.num_special_tokens:
                return token_id
            return token_id + 1  # Shifted by 1 because of the splitter replace token
        return self.bpe_tokenizer.token_to_id(token_str)

    def id_to_token(self, id_: int):
        if self.has_replacement_token:
            if id_ == self.num_special_tokens:
                return self.splitter.replace_token
            elif id_ < self.num_special_tokens:
                return self.bpe_tokenizer.id_to_token(id_)
            else:
                return self.bpe_tokenizer.id_to_token(id_ - 1)
        return self.bpe_tokenizer.id_to_token(id_)

    def encode(self, str_to_encode: str) -> SimpleTokenIdList:
        if self.splitter is not None:
            elemnts_to_encode = self.splitter.split_function(str_to_encode)
            encoded_elements = [self.bpe_tokenizer.encode(element) for element in elemnts_to_encode]  # list[list[int]]
            ret = SimpleTokenIdList()
            for encoded_elements_list in encoded_elements:
                if self.has_replacement_token:
                    to_np = np.array(encoded_elements_list)
                    # add 1 to all non_special tokens since their id in the vocab has been shifted because of the "appended" replace_token
                    ret.ids += (to_np + (to_np >= self.num_special_tokens).astype(int)).tolist()
                    ret.ids.append(self.num_special_tokens)
                else:
                    ret.ids += encoded_elements
            return ret
        return SimpleTokenIdList(self.bpe_tokenizer.encode(str_to_encode).ids)

    def decode(self, ids_list: SimpleTokenIdList) -> str:
        if self.has_replacement_token:
            ret = ""
            elements_list = list_split(ids_list.ids, self.num_special_tokens)
            for i, elem in enumerate(elements_list):
                if i != 0:
                    ret += self.splitter.replace_token
                to_np = np.array(elem)
                to_decode = to_np - (to_np >= self.num_special_tokens).astype(int)
                ret += self.bpe_tokenizer.decode(to_decode.tolist())
            return ret
        return self.bpe_tokenizer.decode(ids_list.ids)
