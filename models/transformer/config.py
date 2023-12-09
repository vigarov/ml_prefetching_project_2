from pathlib import Path


DATASET_PATH="/home/garvalov/ml_prefetching_project_2/data/canneal_v1.csv" # Change depending on which machine is running train.py
INPUT_FEATURES = ["address"]
OUTPUT_FEATURES = ["address"]


def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 22,
        "lr": 10**-4,
        "seq_len": 15,
        "d_model": 512,
        "datasource": 'canneal_trace_v1',
        "data_path": DATASET_PATH,
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "past_window": 10,
        "k_predictions": 10,
        "input_features": INPUT_FEATURES,
        "output_features": OUTPUT_FEATURES
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
