from dataclasses import dataclass, field
import numpy as np

DEBUG_TOKENIZERS = False


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


class Splitter:
    # Ideally, this would be a dataclass; however, lambdas/functions cannot be specified as dataclass fields,
    # as they have no type per se --> normal class
    def __init__(self, split_function, replace_token: str | None = None):
        self.split_function = split_function  # function taking as argument str, returning list(str)
        self.replace_token = replace_token


class SimpleCustomVocabTokenizer:
    def __init__(self, vocab: list[str], special_tokens: list[str], input_splitter: Splitter):
        self.full_vocab = special_tokens + vocab
        self.reverse_dict = {}
        for i, tok in enumerate(self.full_vocab):
            self.reverse_dict[tok] = i
        self.vocab_size = len(self.full_vocab)
        self.input_splitter = input_splitter

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def token_to_id(self, token_str: str) -> int:
        return self.reverse_dict[token_str]

    def id_to_token(self, id_: int) -> str:
        return self.full_vocab[id_]

    def encode(self, str_to_encode: str) -> SimpleTokenIdList:
        all_e = self.input_splitter.split_function(str_to_encode)
        for e in all_e:
            if e not in self.reverse_dict:
                print(e)
                print(str_to_encode)
                print(all_e)
        return SimpleTokenIdList(ids=[self.reverse_dict[element] for element in all_e])

    def decode(self, ids_list: SimpleTokenIdList) -> str:
        return ''.join([self.full_vocab[id_] for id_ in ids_list.ids])

    def get_vocab(self):
        return self.full_vocab


@dataclass
class WrapParameters:
    tokens: tuple[str, str]
    on_encode_wrap_type: str  # "insert_lr", or "no_insert_get_right_index"


class TokenizerWrapper:
    # Important note: this wrapper only preserves the tokenizer's `decode(encode(x)) == x` propery iff the given
    # Splitter's replace_token corresponds to, when added, to the "reverse" of the split_function
    # (e.g.: if split_function == split() and replace_token = " ")
    def __init__(self, bpe_tokenizer, num_special_tokens: int, pad_length: int, splitter: Splitter = None,
                 pad_token: str | None = None, wrap_parameters: WrapParameters | None = None):
        self.num_prepended_extra_special_tokens = 0
        self.wrap_parameters = wrap_parameters
        if wrap_parameters is not None:
            assert len(wrap_parameters.tokens) == 2
            assert wrap_parameters.on_encode_wrap_type in ["insert_lr", "no_insert_get_right_index"]
            self.num_prepended_extra_special_tokens += 2
        self.bpe_tokenizer = bpe_tokenizer
        self.splitter = splitter
        # if not None, we must also be a padder - this is useful for output tokenization, metatransformer, embed_concat
        if pad_token is not None:
            self.num_prepended_extra_special_tokens += 1
        self.pad_token = pad_token
        # if we end up having a splitter replacement token, we want its id to be the last of the special tokens ones
        # (so that pad and unk are always at the same pos, might be important)
        self.num_special_tokens = num_special_tokens
        self.has_replacement_token = self.splitter is not None and self.splitter.replace_token != ""
        self.vocab_size = bpe_tokenizer.get_vocab_size() + self.num_prepended_extra_special_tokens
        if self.has_replacement_token:
            assert self.splitter.replace_token not in self.bpe_tokenizer.get_vocab()
            self.vocab_size += 1
        self.pad_length = pad_length

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def token_to_id(self, token_str: str) -> int:
        if self.pad_token is not None and token_str == self.pad_token:
            return 0
        if self.wrap_parameters is not None:
            internal_start_id = 1 if self.pad_token is not None else 0
            if token_str == self.wrap_parameters.tokens[0]:
                return internal_start_id
            elif token_str == self.wrap_parameters.tokens[1]:
                return internal_start_id + 1
        if self.has_replacement_token:
            if token_str == self.splitter.replace_token:
                return self.num_prepended_extra_special_tokens + self.num_special_tokens
            token_id = self.bpe_tokenizer.token_to_id(token_str)
            if token_id is None:
                raise KeyError
            assert isinstance(token_id, int)
            if token_id < self.num_special_tokens:
                return token_id + self.num_prepended_extra_special_tokens
            return token_id + 1 + self.num_prepended_extra_special_tokens  # Shifted because of the replace, pad tokens
        return self.bpe_tokenizer.token_to_id(token_str)

    def id_to_token(self, id_: int) -> str:
        if self.pad_token is not None:
            if id_ == 0:
                return self.pad_token
            id_ -= 1
        if self.wrap_parameters is not None:
            if id_ == 0:
                return self.wrap_parameters.tokens[0]
            elif id_ == 1:
                return self.wrap_parameters.tokens[1]
            id_ -= 2
        if self.has_replacement_token:
            if id_ == self.num_special_tokens:
                return self.splitter.replace_token
            elif id_ < self.num_special_tokens:
                return self.bpe_tokenizer.id_to_token(id_)
            else:
                return self.bpe_tokenizer.id_to_token(id_ - 1)
        return self.bpe_tokenizer.id_to_token(id_)

    def encode(self, str_to_encode: str) -> (SimpleTokenIdList | tuple[SimpleTokenIdList, int]):
        ret = SimpleTokenIdList()
        if self.splitter is not None:
            elemnts_to_encode = self.splitter.split_function(str_to_encode)
            encoded_elements = [self.bpe_tokenizer.encode(element).ids for element in
                                elemnts_to_encode]  # list[list[int]]
            for i, encoded_elements_list in enumerate(encoded_elements):
                to_np = np.array(encoded_elements_list)
                to_np = to_np + self.num_prepended_extra_special_tokens
                if self.has_replacement_token:
                    replace_token_id = self.token_to_id(self.splitter.replace_token)
                    if i != 0:
                        ret.ids.append(replace_token_id)
                    # add 1 to all non_special tokens since their id in the vocab has been shifted because of the "appended" replace_token
                    # >= and not > because when searching (creating the np bool array), we haven't "added" the replace token
                    # Note: had we not shifted the indices by self.num_extra_... = +2 +1 beforehand, we must've used to_np >= self.num_special_tokens
                    to_np = to_np + (to_np >= replace_token_id).astype(int)

                ret.ids += to_np.tolist()
            # ! must pad if needed before returning
            # Before padding, `return ret` was sufficient here
        else:
            # No need to split, our job is very easy
            raw_tokenization = self.bpe_tokenizer.encode(str_to_encode).ids
            to_np = np.array(raw_tokenization)
            to_np = to_np + self.num_prepended_extra_special_tokens
            ret.ids = to_np.tolist()
        # Before padding, wrap if needed
        if self.wrap_parameters is not None:
            if self.wrap_parameters.on_encode_wrap_type == "insert_lr":
                # left
                ret.ids.insert(0, self.token_to_id(self.wrap_parameters.tokens[0]))
                # right
                ret.ids.append(self.token_to_id(self.wrap_parameters.tokens[1]))
            else:
                assert self.wrap_parameters.on_encode_wrap_type == "no_insert_get_right_index"
                right_index = len(ret.ids)
        if self.pad_token is not None:
            # Pad
            needed_pad_tokens = self.get_pad_length() - len(ret.ids)
            if needed_pad_tokens < 0:
                raise ValueError(f"Input too long, tokenized into {len(ret.ids)}, yet max is {self.get_pad_length()}")
            ret.ids = ret.ids + [self.token_to_id(self.pad_token)] * needed_pad_tokens
        if self.wrap_parameters is not None and self.wrap_parameters.on_encode_wrap_type == "no_insert_get_right_index":
            return ret, right_index
        else:
            return ret

    def decode(self, ids_list: SimpleTokenIdList) -> str:
        # When decoding, the only special thing we need to do for padding is ignore all `pad` tokens
        to_np = np.array(ids_list.ids)
        if self.pad_token is not None:
            to_np = to_np[to_np != self.token_to_id(self.pad_token)]
        if self.wrap_parameters is not None:
            # Remove SOS/EOS ;
            to_np = to_np[np.all(
                [to_np != self.token_to_id(self.wrap_parameters.tokens[0]),
                 to_np != self.token_to_id(self.wrap_parameters.tokens[0])],
                axis=0)]
        to_np = to_np - self.num_prepended_extra_special_tokens
        ids_list.ids = to_np.tolist()
        if self.has_replacement_token:
            ret = ""
            # Conceptually speaking, we want to do
            # elements_list = list_split(ids_list.ids, self.token_to_id(self.splitter.replace_token))
            # However, since we've already shifted the ids based on the extra features, we must adapt the split token:
            elements_list = list_split(ids_list.ids,
                                       self.token_to_id(
                                           self.splitter.replace_token) - self.num_prepended_extra_special_tokens)
            for i, elem in enumerate(elements_list):
                if i != 0:
                    ret += self.splitter.replace_token
                to_np = np.array(elem)
                to_decode = to_np - (to_np >= self.num_special_tokens).astype(int)
                ret += self.bpe_tokenizer.decode(to_decode.tolist())
            return ret
        return self.bpe_tokenizer.decode(ids_list.ids)

    def get_pad_length(self):
        return self.pad_length

    def wraps(self):
        return self.wrap_parameters is not None

    def returns_right_wrap_index(self):
        assert self.wraps()
        return self.wrap_parameters.on_encode_wrap_type == "no_insert_get_right_index"

    def has_already_padded(self):
        return self.pad_token is not None


class ConcatTokenizer:
    def __init__(self, feature_separator_token: str, pad_token: str, tokenizers: list[TokenizerWrapper]):
        self.pad_token, self.pt_id = pad_token, 0
        self.feature_separator_token, self.fst_id = feature_separator_token, 1
        self.tokenizers = list(tokenizers)
        vocab_cumsums = np.cumsum(
            [tokenizer.get_vocab_size() for tokenizer in self.tokenizers]) + 2  # +2 to account for the fsp,pad tokens
        self.vocab_size = int(vocab_cumsums[-1])
        vocab_cumsums = vocab_cumsums.tolist()
        vocab_cumsums.insert(0, 0)  # la tete a Toto
        self.vocab_cumsums = vocab_cumsums
        for tokenizer in tokenizers:
            assert not tokenizer.wraps() or not tokenizer.returns_right_wrap_index()

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def token_to_id(self, token_str: str) -> list[int]:
        # Returns a list of all possible ids, as a token might be present in > 1 (sub) tokenizer
        if token_str == self.pad_token:
            return [self.pt_id]
        elif token_str == self.feature_separator_token:
            return [self.fst_id]
        all_possible_ids = []
        for tokenizer in self.tokenizers:
            ret = tokenizer.token_to_id(token_str)
            if ret and ret != "":
                all_possible_ids.append(ret)
        return all_possible_ids

    def id_to_token(self, id_: int) -> str:
        assert id_ >= 0
        if id_ == 0:
            return self.pad_token
        elif id == 1:
            return self.feature_separator_token
        else:
            for i, s in enumerate(self.vocab_cumsums):
                if id_ < s:
                    return self.tokenizers[i - 1].id_to_token(id_)
            raise LookupError

    def encode(self, features_to_encode: list[str], pad=True) -> SimpleTokenIdList:
        assert len(features_to_encode) == len(self.tokenizers)
        ret = SimpleTokenIdList()
        for i, feature_value in enumerate(features_to_encode):
            tokenizer = self.tokenizers[i]
            encoded_tokens_list = (np.array(tokenizer.encode(feature_value).ids) +
                                   self.vocab_cumsums[i]).tolist()
            if pad and not tokenizer.has_already_padded():
                needed_pad_tokens = tokenizer.get_pad_length() - len(encoded_tokens_list)
                if needed_pad_tokens < 0:
                    raise ValueError(
                        f"Input too long for feature {i}: max length permitted = {tokenizer.get_pad_length()}, while the encoded sequence was {len(encoded_tokens_list)}")
                encoded_tokens_list = encoded_tokens_list + [self.pt_id] * needed_pad_tokens
            ret.ids += encoded_tokens_list
            if i != len(features_to_encode) - 1:
                ret.ids += [self.fst_id]  # Add feature separator token
        return ret

    def decode(self, ids_list: SimpleTokenIdList) -> list[str]:
        # Will never be used in our model, as for the output (which is the only place we'll need `decode()`), we only
        # have one feature --> will not be wrapped around ConcatTokenizer
        if not DEBUG_TOKENIZERS:
            raise PermissionError

        # An implementation of the method can still be found down below though:
        # Note: this might not take care of padding, can't be bothered to check
        per_tokenizer_lists = list(list_split(ids_list.ids, 0))
        assert len(per_tokenizer_lists) == len(self.tokenizers)
        ret = []
        for i, tokenizer_list in enumerate(per_tokenizer_lists):
            tokenizer_list = (np.array(tokenizer_list) - self.vocab_cumsums[i]).to_list()
            out_str = self.tokenizers[i].decode(tokenizer_list)
            ret.append(out_str)
        return ret
