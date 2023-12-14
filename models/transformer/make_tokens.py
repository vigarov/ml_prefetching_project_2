from config import get_config, FORCE_BYTE
from pandas import read_csv, concat
from itertools import groupby

# Huggingface tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import BPEDecoder

from pathlib import Path
import numpy as np

MIN_FREQUENCY = 4


def build_bpe_tokenizer(config, df, feature_type,feature_list):
    """
        Train a different BPE tokenizer for all feature types in the data frame
        features_list is the list of all the features with the same type
    """
    print(f"Creating Tokenizer for feature type {feature_type}")
    tokenizer_path = Path(config['tokenizer_files'].format(feature_type))
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    create_tokenizer = not Path.exists(tokenizer_path)
    if not create_tokenizer:
        ow = input(f"tokenizer already exists for feature '{feature_type}', overwrite ?").lower()
        create_tokenizer = ow == "y" or ow == "yes"
    if create_tokenizer:
        special_tokens = config["bpe_special_tokens"]
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()  # doesn't matter, but helps for training!
        trainer = BpeTrainer(special_tokens=special_tokens,min_frequency=MIN_FREQUENCY)
        iterable = df[[feature.name for feature in feature_list]].to_numpy().flatten()
        tokenizer.train_from_iterator(iterable, trainer=trainer)
        if "hex" in feature_type and FORCE_BYTE:
            # will force to learn all bytes in vocab, good to reduce model size at low vocab increase cost :D
            tokenizer.add_tokens(
                list({hex(i)[2:].zfill(2) for i in range(int("0xff", 16))} - set(tokenizer.get_vocab().keys())))
        tokenizer.save(str(Path(tokenizer_path).absolute().as_posix()))


def get_data_frame(path, use_all_directory=True):
    input_path = Path(path)
    assert input_path.exists()
    if not use_all_directory:
        assert input_path.is_file() and input_path.suffix == ".csv"
        df = read_csv(input_path.as_posix())
    else:
        if input_path.is_file():
            input_path = input_path.parent
        assert input_path.is_dir()
        files_in_dir = [x for x in input_path.iterdir() if x.is_file() and x.suffix == ".csv"]
        assert len(files_in_dir) > 0
        dfs = [read_csv(f.absolute().as_posix()) for f in files_in_dir]
        df = concat(dfs)
    df = df.astype({col: str for col in df.columns if df[col].dtype == "int64"})
    return df


if __name__ == "__main__":
    config = get_config()
    df = get_data_frame(config["data_path"], use_all_directory=True)
    assert "int64" not in df.dtypes
    # we don't care about lists for building, the wrappers will do the "hard work"
    # when we instantiate the tokenizers before training
    feature_key = lambda feature: feature.get_primitive_type()
    gb_generator = groupby(sorted(config["input_features"] + config["output_features"], key=feature_key),
                           feature_key)
    for primitive_feature_type,feature_list in gb_generator:
        feature_list = list(feature_list)
        assert len(feature_list) > 0
        build_bpe_tokenizer(config, df, primitive_feature_type,feature_list)
