"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import math
import random
import re
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from nltk import edit_distance
from pytorch_lightning.utilities import rank_zero_only
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from donut import DonutConfig, DonutModel

class DonutModelPLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        print("DEBUG: Initializing DonutModelPLModule")
        if self.config.get("pretrained_model_name_or_path", False):
            self.model = DonutModel.from_pretrained(
                self.config.pretrained_model_name_or_path,
                input_size=self.config.input_size,
                max_length=self.config.max_length,
                align_long_axis=self.config.align_long_axis,
                ignore_mismatched_sizes=True,
            )
        else:
            self.model = DonutModel(
                config=DonutConfig(
                    input_size=self.config.input_size,
                    max_length=self.config.max_length,
                    align_long_axis=self.config.align_long_axis,
                )
            )
        self.pytorch_lightning_version_is_1 = int(pl.__version__[0]) < 2
        self.num_of_loaders = len(self.config.dataset_name_or_paths)
        print("DEBUG: DonutModelPLModule initialized")

    def training_step(self, batch, batch_idx):
        print("DEBUG: Entering training_step")
        # Unpack the batch
        image_tensors, decoder_input_ids, decoder_labels = zip(*batch)
        image_tensors = image_tensors[0]
        decoder_input_ids = decoder_input_ids[0]
        decoder_labels = decoder_labels[0]

        # Process input tensors
        decoder_input_ids = decoder_input_ids[:, :-1]
        decoder_labels = decoder_labels[:, 1:]

        # Calculate loss
        loss = self.model(image_tensors, decoder_input_ids, decoder_labels)[0]
        print("DEBUG: Training step loss calculated")

        # Log the loss
        self.log_dict({"train_loss": loss}, sync_dist=True)
        if not self.pytorch_lightning_version_is_1:
            self.log('loss', loss, prog_bar=True)

        return loss

    def on_validation_epoch_start(self):
        print("DEBUG: Validation epoch started")
        super().on_validation_epoch_start()
        self.validation_step_outputs = [[] for _ in range(self.num_of_loaders)]
        return

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        print("DEBUG: Entering validation_step")
        image_tensors, decoder_input_ids, prompt_end_idxs, answers = batch
        decoder_prompts = pad_sequence(
            [input_id[: end_idx + 1] for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)],
            batch_first=True,
        )

        preds = self.model.inference(
            image_tensors=image_tensors,
            prompt_tensors=decoder_prompts,
            return_json=False,
            return_attentions=False,
        )["predictions"]

        scores = list()
        for pred, answer in zip(preds, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            answer = re.sub(r"<.*?>", "", answer, count=1)
            answer = answer.replace(self.model.decoder.tokenizer.eos_token, "")
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"DEBUG: Prediction: {pred}")
                print(f"DEBUG: Answer: {answer}")
                print(f"DEBUG: Normalized Edit Distance: {scores[0]}")

        self.validation_step_outputs[dataloader_idx].append(scores)
        print("DEBUG: Validation step completed")

        return scores

    def on_validation_epoch_end(self):
        print("DEBUG: Validation epoch ended")
        assert len(self.validation_step_outputs) == self.num_of_loaders
        cnt = [0] * self.num_of_loaders
        total_metric = [0] * self.num_of_loaders
        val_metric = [0] * self.num_of_loaders
        for i, results in enumerate(self.validation_step_outputs):
            for scores in results:
                cnt[i] += len(scores)
                total_metric[i] += np.sum(scores)
            val_metric[i] = total_metric[i] / cnt[i]
            val_metric_name = f"val_metric_{i}th_dataset"
            self.log_dict({val_metric_name: val_metric[i]}, sync_dist=True)
        self.log_dict({"val_metric": np.sum(total_metric) / np.sum(cnt)}, sync_dist=True)
        print("DEBUG: Validation metrics logged")

    def configure_optimizers(self):
        print("DEBUG: Configuring optimizers")
        max_iter = None

        if int(self.config.get("max_epochs", -1)) > 0:
            assert len(self.config.train_batch_sizes) == 1, "Set max_epochs only if the number of datasets is 1"
            max_iter = (self.config.max_epochs * self.config.num_training_samples_per_epoch) / (
                self.config.train_batch_sizes[0] * torch.cuda.device_count() * self.config.get("num_nodes", 1)
            )

        if int(self.config.get("max_steps", -1)) > 0:
            max_iter = min(self.config.max_steps, max_iter) if max_iter is not None else self.config.max_steps

        assert max_iter is not None
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        scheduler = {
            "scheduler": self.cosine_scheduler(optimizer, max_iter, self.config.warmup_steps),
            "name": "learning_rate",
            "interval": "step",
        }
        print("DEBUG: Optimizers configured")
        return [optimizer], [scheduler]

    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        print("DEBUG: Entering cosine_scheduler")
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)
class DonutDataPLModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_batch_sizes = self.config.train_batch_sizes
        self.val_batch_sizes = self.config.val_batch_sizes
        self.train_datasets = []
        self.val_datasets = []
        self.g = torch.Generator()
        self.g.manual_seed(self.config.seed)
        print("DEBUG: DonutDataPLModule initialized")

    def train_dataloader(self):
        print("DEBUG: Creating train dataloader")
        loaders = list()
        for train_dataset, batch_size in zip(self.train_datasets, self.train_batch_sizes):
            loaders.append(
                DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    num_workers=self.config.num_workers,
                    pin_memory=True,
                    worker_init_fn=self.seed_worker,
                    generator=self.g,
                    shuffle=True,
                    # collate_fn=custom_collate_fn,
                )
            )
        print("DEBUG: Train dataloaders created")
        return loaders

    def val_dataloader(self):
        print("DEBUG: Creating validation dataloader")
        loaders = list()
        for val_dataset, batch_size in zip(self.val_datasets, self.val_batch_sizes):
            loaders.append(
                DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    pin_memory=True,
                    shuffle=False,
                    # collate_fn=custom_collate_fn,
                )
            )
        print("DEBUG: Validation dataloaders created")
        return loaders

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        print(f"DEBUG: Seed worker initialized with worker_id: {worker_id}")


