from config import get_config

from pandas import DataFrame

# Huggingface tokenizers
from models.transformer.tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import  BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path


def get_all_features(ds,feature_name):
    for item in ds:
        yield item[feature_name]

def build_tokenizer(config, ds, feature_name):
    """
        Train a different BPE tokenizer for all "feature_name" in the data set ds
    """
    tokenizer_path = Path(config['tokenizer_file'].format(feature_name))
    create_tokenizer = not Path.exists(tokenizer_path)
    if not create_tokenizer:
        ow = input("tokenizer already exists, overwrite ?").lower()
        create_tokenizer = ow == "y" || ow == "yes"
    if create_tokenizer:
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=["[UNK]","[PAD]"], min_frequency=3)
        tokenizer.train_from_iterator(get_all_features(ds,feature_name), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    

if __name__ == "__main__":
    config = get_config()
    ds = DataFrame.from_csv(config["data_path"])
    for feature in config["input_features"]:
        build_tokenizer(config,ds,feature)