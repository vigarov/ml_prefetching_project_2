from config import get_config
from pandas import read_csv, concat
from itertools import groupby

# Huggingface tokenizers
from tokenizers import Tokenizer, PreTokenizedString
from tokenizers.models import BPE
from tokenizers.trainers import  BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, PreTokenizer

from pathlib import Path

def build_tokenizer(config, df, features_list):
    """
        Train a different BPE tokenizer for all feature types in the data frame
        features_list is the list of all the features with the same type
    """
    feature_type = features_list[0].type
    print(f"Creating Tokenizer for feature type {feature_type}")
    tokenizer_path = Path(config['tokenizer_file'].format(feature_type))
    create_tokenizer = not Path.exists(tokenizer_path)
    if not create_tokenizer:
        ow = input(f"tokenizer already exists for feature '{feature_type}', overwrite ?").lower()
        create_tokenizer = ow == "y" or ow == "yes"
    if create_tokenizer:
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=["[UNK]","[PAD]"], min_frequency=4)
        iterable = df[[feature.name for feature in feature_list]].to_numpy().flatten()
        tokenizer.train_from_iterator(iterable, trainer=trainer)
        tokenizer.save(str(tokenizer_path))

def get_data_frame(path,use_all_directory=True):
    input_path = Path(path)
    assert input_path.exists()
    if not use_all_directory:
        assert input_path.is_file() and input_path.suffix == ".csv"
        df = read_csv(input_path.as_posix())
    else:
        if input_path.is_file():
            input_path = input_path.parent
        assert input_path.is_dir()
        files_in_dir = [x for x in input_path.iterdir() if x.is_file() and x.suffix==".csv"]
        assert len(files_in_dir) > 0
        dfs = [read_csv(f.absolute().as_posix()) for f in files_in_dir]
        df = concat(dfs)
    df = df.astype({col: str for col in df.columns if df[col].dtype == "int64"})
    return df

if __name__ == "__main__":
    config = get_config()
    df = get_data_frame(config["data_path"],use_all_directory=True)
    assert "int64" not in df.dtypes
    feature_key = lambda feature: feature.type
    features_by_type = [list(group) for key, group in groupby(sorted(config["input_features"]+config["output_features"],key=feature_key), feature_key)]
    for feature_list in features_by_type:
        assert len(feature_list) > 0 
        build_tokenizer(config,df,feature_list)
