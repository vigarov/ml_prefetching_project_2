from config import get_config
from pandas import read_csv

# Huggingface tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import  BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def build_tokenizer(config, df, feature):
    """
        Train a different BPE tokenizer for all "feature_name" in the data set ds
    """
    tokenizer_path = Path(config['tokenizer_file'].format(feature.type))
    create_tokenizer = not Path.exists(tokenizer_path)
    if not create_tokenizer:
        ow = input("tokenizer already exists, overwrite ?").lower()
        create_tokenizer = ow == "y" or ow == "yes"
    if create_tokenizer:
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=["[UNK]","[PAD]"], min_frequency=4)
        tokenizer.train_from_iterator(df[feature.name].copy(), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    

if __name__ == "__main__":
    config = get_config()
    ds = read_csv(config["data_path"])
    for feature in config["input_features"]:
        build_tokenizer(config,ds,feature)