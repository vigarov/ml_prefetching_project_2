import warnings
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

import models.common.config as conf
from models.transformer.train import train_model as train_transformer, get_ds, get_model
from models.retnet.train import train_model as train_retnet

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = conf.get_config()
    # Define the device
    device = conf.get_device()
    # torch.set_default_device(device)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = conf.latest_weights_file_path(config) if preload == 'latest' else conf.get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')
    match config['attention_model']:
        case 'transformer':
            train_transformer(model, config, optimizer, train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt,
                              device, writer, initial_epoch, global_step)
        case 'retnet':
            train_retnet(model, config, train_dataloader, val_dataloader, tokenizer_src)




