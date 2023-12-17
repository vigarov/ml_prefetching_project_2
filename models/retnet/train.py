import torch
from transformers import (Trainer, TrainingArguments, DataCollatorForLanguageModeling)
from transformers.data.data_collator import DataCollatorMixin, DefaultDataCollator

from models.common.config import get_weights_file_path


def train_model(model, config, train_dataset, eval_dataset, tokenizer, loss_fct):
    model_filename = get_weights_file_path(config, "retnet")
    train_args = TrainingArguments(num_train_epochs=config['num_epochs'], output_dir=model_filename,
                                   save_strategy="epoch", overwrite_output_dir=True, evaluation_strategy="epoch",
                                   learning_rate=config['lr'])

    print(train_args.device)

    trainer = CustomTrainer(model=model,
                            args=train_args,
                            train_dataset=train_dataset.dataset,
                            eval_dataset=eval_dataset.dataset,
                            tokenizer=tokenizer,
                            loss_fct=loss_fct,
                            data_collator=EmptyCollator())

    trainer.train()
    return trainer.model


class CustomTrainer(Trainer):

    def __init__(self, loss_fct, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = loss_fct

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss = self.loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class EmptyCollator(DataCollatorMixin):
    def __call__(self, features, return_tensors=None):
        return {"labels": torch.stack([f["label"] for f in features])}
