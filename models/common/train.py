import warnings
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer

import models.common.config as conf
from models.transformer.train import train_model as train_transformer, get_ds, get_model
from models.retnet.train import train_model as train_retnet

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = conf.get_config(model="retnet")
    # Define the device
    device = conf.get_device()

    # Check if we're in the middle of a "train" : if yes, load the current random sample (based off the seed)
    # Else, take (new) random seed
    was_currently_training = len(conf.source_model_files(config)) != 0
    generator = torch.Generator(device)
    mp = Path(conf.get_model_full_path(config))
    # Make sure the model's run folder exists
    mp.mkdir(parents=True, exist_ok=True)
    # Also create the generator folder
    gen_dir = mp / conf.GENERATOR_PREFIX
    gen_dir.mkdir(parents=True, exist_ok=True)
    if was_currently_training:
        seed_path = gen_dir / conf.SEED_FN
        assert seed_path.exists()
        seed = torch.load(seed_path.absolute().as_posix())
        generator = generator.manual_seed(seed)
        state_path = gen_dir / conf.STATE_FN
        assert state_path.exists()
        gen_state = torch.load(state_path.absolute().as_posix())
    else:
        torch.save(generator.seed(), (gen_dir / conf.SEED_FN).as_posix())

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config, generator)
    # Save generator TODO: useless to do here, save at weight save time iff generator impacts training
    torch.save(generator.get_state(), (gen_dir / conf.STATE_FN).as_posix())
    model = get_model(config, tokenizer_src, tokenizer_tgt).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_folder'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = conf.latest_weights_file_path(config) if preload == 'latest' \
        else conf.get_weights_file_path(config, preload) if preload \
        else None
    if model_filename:
        assert was_currently_training
        generator.set_state(
            gen_state)  # TODO: I think we might not need that - past weight initialization, there is no randomness right? Incorrect for dropout? Incorrect for beam search?
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    embedding_type = config["embedding_technique"]
    if embedding_type in ["meta_transformer", "embed_concat"]:
        # TODO: one loss per embedding layer? imo yes
        raise NotImplementedError
    else:
        input_src_potential_pad_ids = tokenizer_src.token_to_id(config["pad_token"])
        output_pad_id = tokenizer_tgt.token_to_id(config["pad_token"])
        assert len(input_src_potential_pad_ids) == 1 and input_src_potential_pad_ids[0] == output_pad_id
        loss_fn = nn.CrossEntropyLoss(ignore_index=output_pad_id,
                                      label_smoothing=0.1).to(device)
    match config['attention_model']:
        case 'transformer':
            train_transformer(model, config, optimizer, train_dataloader, val_dataloader, loss_fn, tokenizer_tgt,
                              device, writer, initial_epoch, global_step)
        case 'retnet':
            train_retnet(model, config, train_dataloader, val_dataloader, tokenizer_src, loss_fn)




