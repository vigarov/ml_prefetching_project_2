from pathlib import Path
from dataclasses import dataclass
import re

DATA_PATH = "data/processed/"
SEED_FN = "rand_seed.txt"
STATE_FN = "state.pt"


@dataclass
class Feature:
    name: str
    type: str
    max_len: int # in tokens

    def __str__(self):
        return self.name


PAST_WINDOW = 10
K_PREDICTIONS = 10
FORCE_BYTE = True

# max_len for prev_pfaults is at most "past_window" * 17 +2 = address_size_in_hex_digits + 1 ("0x") + 2 (potential EOS/SOS) or 9 if FORCE_BYTE
#             bitmap is 16 + 2 (EOS/SOS)
#             ip, 16+2, similar as prev_pfaults, but no array-> no mult
#             ustack, don't know yet, use big value for now TODO
#             regs, #regs*len_max_reg_value_in_chosen_base = 20 * (16 + 1 ("0x")) + 2 (EOS/SOS) = 20 * (8+1) + 2 if FORCE_BYTE

HEX_64_LEN = (8 if FORCE_BYTE else 16) + 1  # 1 for "0x"

INPUT_FEATURES = [Feature("prev_faults", "hex_address",PAST_WINDOW*HEX_64_LEN + 2), Feature("flags", "bitmap",18), Feature("ip", "hex_address",HEX_64_LEN+2),
                  Feature("ustack", "text",250), Feature("regs", "hex_number",20*HEX_64_LEN+2)]
OUTPUT_FEATURES = [Feature("y", "hex_address",K_PREDICTIONS*HEX_64_LEN+2)]

@dataclass
class TransformerModelParams:
    d_model: int = 512
    T: int = 6  # Num Transformer blocks layers
    H: int = 8  # Num Attention heads per Transformer layer
    dropout: float = 0.1
    d_ff: int = 2048

def get_config():
    config = {
        "special_tokens": ["[PAD]","[SPR]","[UNK]"],  # Global
        "batch_size": 8,  # Training hyperparameter
        "num_epochs": 22,  # Training hyperparameter
        "lr": 10 ** -4,  # Training hyperparameter
        "datasource": "canneal",  # Global
        "model_folder": "models",  # Global
        "preload": "latest",  # Global
        "tokenizer_files": "tokenizers/tokenizer_{0}.json",  # Global
        "experiment_name": "runs/tmodel",  # Global
        "train_test_split": 0.75,  # Training hyperparameter
        "attention_model": "transformer",  # Model hyperparameter, choose with "retnet"
        "attention_model_params" : TransformerModelParams(),  # Model hyperparameter
        "past_window": PAST_WINDOW,  # Model hyperparameter
        "k_predictions": K_PREDICTIONS,  # Model hyperparameter
        "input_features": INPUT_FEATURES,  # Model hyperparameter
        "output_features": OUTPUT_FEATURES,  # Model hyperparameter
        "embedding_technique": "concat_tokens"  # Model hyperparameter, choose with "hextet_concat", "onetext", "meta_transofrmer" TODO:, "embed_concat"
    }

    max_path = None
    max_path_version = 0.
    for item in Path(DATA_PATH).iterdir():
        if item.is_file():
            matches = re.search(config["datasource"] + r"_v(\d+\.\d+)", item.name)
            if matches is not None:
                # We have a file that corresponds to data_source
                version = matches.group(1)
                if float(version) > max_path_version:
                    max_path = item
                    max_path_version = float(version)
    assert max_path is not None and max_path_version > 0

    config["data_path"] = max_path.absolute().as_posix()

    model_hash_features = ["attention_model", "past_window", "k_predictions", "input_features",
                           "embedding_technique","embedding_technique"]

    def parse_mhf(feature_name):
        model_feature = config[feature_name]
        assert model_feature is not None
        if isinstance(model_feature, list):
            return "".join([s.name[:1] for s in model_feature])
        return str(model_feature)

    config["model_basename"] = "_".join(parse_mhf(mhf)[:5] for mhf in model_hash_features)

    return config


def get_model_full_path(config):
    return f"trainings/{config['datasource']}/{config['model_basename']}/"


def get_weights_file_path(config, epoch: str):
    p = Path(get_model_full_path(config))
    weights_filename = f"weights{epoch}.pt"
    return str(p / weights_filename)


def source_model_files(config):
    return list(Path(get_model_full_path(config)).glob('*'))


# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    weights_files = source_model_files(config)
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
