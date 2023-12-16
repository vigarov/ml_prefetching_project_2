from transformers import (Trainer, TrainingArguments)

from models.common.config import get_weights_file_path


def train_model(model, config, train_dataset, eval_dataset, tokenizer):
    model_filename = get_weights_file_path(config, "retnet")
    train_args = TrainingArguments(num_train_epochs=config['num_epochs'], output_dir=model_filename,
                                   save_strategy="epoch", overwrite_output_dir=True, evaluation_strategy="epoch",
                                   learning_rate=config['lr'])

    print(train_args.device)


    trainer = Trainer(model=model,
                      args=train_args,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      tokenizer=tokenizer)

    trainer.train()
    return trainer.model
