# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/training.ipynb (unless otherwise specified).

__all__ = ['dynamic_padding', 'TransformerTrainer']

# Cell
from .translation import translate_sentence
from .metrics import calculate_bleu_score
from tqdm import tqdm

import torch.nn as nn
import numpy as np
import torch
import wandb
import os

# Cell
def dynamic_padding(sentences, pad_token_id=3):
    first_padding_id = np.argmax(sentences["input_ids"].numpy() == pad_token_id, axis=1).max()

    if first_padding_id == 0:
        return sentences

    sentences["input_ids"] = sentences["input_ids"][:, :first_padding_id]
    sentences["padding_mask"] = sentences["padding_mask"][:, :first_padding_id]

    return sentences


# Cell
class TransformerTrainer:

    def __init__(self, model, criterion, optimizer, allow_dynamic_padding, lr_scheduler=None, calc_bleu=False, src_tokenizer=None, trg_tokenizer=None, pad_token_id=None, device="cuda", wandb_log=False):

        assert calc_bleu and src_tokenizer is not None and trg_tokenizer is not None, "Provide src_tokenizer and trg_tokenizer for calculating BLEU score"

        self.calc_bleu = calc_bleu
        self.wandb_log = wandb_log
        self.pad_token_id = pad_token_id
        self.allow_dynamic_padding = allow_dynamic_padding

        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer

        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def log_to_wandb(self, step, stage, loss=None, bleu=None):

        if stage == "test" and self.calc_bleu:
            wandb.log({f"{stage}/{stage}_bleu": bleu})

            return

        metrics = {
            f"{stage}/step": step,
            **{f"{stage}/{stage}_loss": loss},
        }

        if stage == "train":
            metrics[f"{stage}/{stage}_lr"] = self.optimizer.get_current_lr()

        wandb.log(metrics)

    @torch.no_grad()
    def calc_batch_bleu(self, batch):
        batch_bleu = 0
        for src_sentence, trg_sentence in zip(batch["src_sentence"]["text"], batch["trg_sentence"]["text"]):
            batch_bleu += calculate_bleu_score(self.model, src_sentence, trg_sentence, self.src_tokenizer, self.trg_tokenizer)

        return batch_bleu / len(batch["src_sentence"]["text"])

    def train_one_epoch(self, epoch, dataloader):
        self.model.train()

        total_steps = len(dataloader)
        total_train_loss = 0

        with tqdm(enumerate(dataloader, 1), unit="batch", total=total_steps, bar_format='{l_bar}{bar:10}{r_bar}') as progress_bar:
            progress_bar.set_description(f"Epoch {epoch+1}".ljust(25))
            for step, batch in progress_bar:

                total_train_loss += self.train_one_step(batch)
                current_loss = total_train_loss / step

                progress_bar.set_postfix({"train loss": f"{current_loss:.3f}"})

                if self.wandb_log:
                    current_step = epoch * total_steps + step
                    self.log_to_wandb(current_step, "train", loss=current_loss)

        total_train_loss /= total_steps

        return total_train_loss

    @torch.no_grad()
    def validate_one_epoch(self, epoch, dataloader):
        self.model.eval()

        total_valid_loss = 0
        total_steps = len(dataloader)
        with tqdm(enumerate(dataloader, 1), unit="batch", total=total_steps, bar_format='{l_bar}{bar:10}{r_bar}') as progress_bar:
            progress_bar.set_description(f"Validation after epoch {epoch+1}".ljust(25))

            for step, batch in progress_bar:
                total_valid_loss += self.validate_one_step(batch)

                progress_bar.set_postfix({"valid loss": f"{(total_valid_loss / step):.3f}"})

        total_valid_loss = total_valid_loss / total_steps

        if self.wandb_log:
            self.log_to_wandb(epoch, "valid", loss=total_valid_loss)

        return total_valid_loss

    def process_batch(self, batch):
        if self.allow_dynamic_padding:
            batch["src_sentence"] = dynamic_padding(batch["src_sentence"], pad_token_id=self.pad_token_id)
            batch["trg_sentence"] = dynamic_padding(batch["trg_sentence"], pad_token_id=self.pad_token_id)

        src_input_ids = batch["src_sentence"]["input_ids"].to(self.device)
        src_mask = batch["src_sentence"]["padding_mask"].to(self.device)

        trg_input_ids = batch["trg_sentence"]["input_ids"].to(self.device)
        trg_mask = batch["trg_sentence"]["padding_mask"].to(self.device)

        return src_input_ids, src_mask, trg_input_ids, trg_mask

    def train_one_step(self, batch):
        src_input_ids, src_mask, trg_input_ids, trg_mask = self.process_batch(batch)

        trg_model_input_ids = trg_input_ids[:, :-1]
        trg_model_input_mask = trg_mask[:, :-1]

        trg_ground_truth_input_ids = trg_input_ids[:, 1:].flatten()

        outputs = self.model(src_input_ids, trg_model_input_ids, src_mask, trg_model_input_mask)
        outputs = outputs.view(-1, outputs.shape[-1])

        batch_loss = self.criterion(outputs, trg_ground_truth_input_ids)
        batch_loss.backward()

        self.optimizer.step()


        for p in self.model.parameters():
            p.grad = None

        return batch_loss.item()


    def validate_one_step(self, batch):
        src_input_ids, src_mask, trg_input_ids, trg_mask = self.process_batch(batch)

        trg_model_input_ids = trg_input_ids[:, :-1]
        trg_model_input_mask = trg_mask[:, :-1]

        trg_ground_truth_input_ids = trg_input_ids[:, 1:].flatten()

        outputs = self.model(src_input_ids, trg_model_input_ids, src_mask, trg_model_input_mask)
        outputs = outputs.view(-1, outputs.shape[-1])

        batch_loss = self.criterion(outputs, trg_ground_truth_input_ids)

        return batch_loss.item()

    def fit(self, epochs, train_dataloader, valid_dataloader, test_dataloader=None, test_step=1, save_path="trained_models"):

        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_path += f"/best_state_dict.pth"

        best_valid_loss = np.inf

        for epoch in range(epochs):
            epoch_loss = self.train_one_epoch(epoch, train_dataloader)
            valid_loss = self.validate_one_epoch(epoch, valid_dataloader)

            if self.calc_bleu and test_dataloader is not None and epoch % test_step == 0 and epoch > 0:
                self.model.eval()

                total_bleu = 0
                with tqdm(enumerate(test_dataloader, 1), unit="batch", total=len(test_dataloader), bar_format='{l_bar}{bar:10}{r_bar}') as progress_bar:
                    progress_bar.set_description(f"BLEU after epoch {epoch+1}".ljust(25))
                    for step, batch in progress_bar:
                        total_bleu += self.calc_batch_bleu(batch)
                        progress_bar.set_postfix({"bleu": f"{total_bleu / step:.3f}"})

                total_bleu = total_bleu / len(test_dataloader)

                if self.wandb_log:
                    self.log_to_wandb(epoch, "test", bleu=total_bleu)

            if self.lr_scheduler is not None: self.lr_scheduler.step(valid_loss)

            if save_path:
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(self.model.state_dict(), save_path)

        return best_valid_loss