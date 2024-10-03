"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import argparse
import datetime
import json
import os
import random
from io import BytesIO
from os.path import basename
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.utilities import rank_zero_only
from sconf import Config

from donut import DonutDataset
from lightning_module import DonutDataPLModule, DonutModelPLModule



class CustomCheckpointIO(CheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        del checkpoint["state_dict"]
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, storage_options=None):
        checkpoint = torch.load(path + "artifacts.ckpt")
        state_dict = torch.load(path + "pytorch_model.bin")
        checkpoint["state_dict"] = {"model." + key: value for key, value in state_dict.items()}
        return checkpoint

    def remove_checkpoint(self, path) -> None:
        return super().remove_checkpoint(path)


@rank_zero_only
def save_config_file(config, path):
    if not Path(path).exists():
        os.makedirs(path)
    save_path = Path(path) / "config.yaml"
    print(config.dumps())
    with open(save_path, "w") as f:
        f.write(config.dumps(modified_color=None, quote_str=True))
        print(f"Config is saved at {save_path}")


class ProgressBar(pl.callbacks.TQDMProgressBar):
    def __init__(self, config):
        super().__init__()
        self.enable = True
        self.config = config

    def disable(self):
        self.enable = False

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        items["exp_name"] = f"{self.config.get('exp_name', '')}"
        items["exp_version"] = f"{self.config.get('exp_version', '')}"
        return items


def set_seed(seed):
    pytorch_lightning_version = int(pl.__version__[0])
    if pytorch_lightning_version < 2:
        pl.utilities.seed.seed_everything(seed, workers=True)
    else:
        import lightning_fabric
        lightning_fabric.utilities.seed.seed_everything(seed, workers=True)


def train(config):
    print("DEBUG: Starting training")
    set_seed(config.get("seed", 42))

    model_module = DonutModelPLModule(config)
    data_module = DonutDataPLModule(config)

    # Add datasets to data_module
    datasets = {"train": [], "validation": []}
    for i, dataset_name_or_path in enumerate(config.dataset_name_or_paths):
        task_name = os.path.basename(dataset_name_or_path)
        print(f"DEBUG: Loading dataset {task_name}")

        common_tokens = [
            # General figure information
            "<s_general_figure_info>", "</s_general_figure_info>",
            "<s_figure_info>", "</s_figure_info>",
            "<s_bbox>", "</s_bbox>", "<s_x>", "</s_x>", "<s_y>", "</s_y>", "<s_w>", "</s_w>", "<s_h>", "</s_h>",

            # Axes information
            "<s_y_axis>", "</s_y_axis>",
            "<s_y_axis_label>", "</s_y_axis_label>",
            "<s_y_axis_major_labels>", "</s_y_axis_major_labels>",
            "<s_x_axis>", "</s_x_axis>",
            "<s_x_axis_label>", "</s_x_axis_label>",
            "<s_x_axis_major_labels>", "</s_x_axis_major_labels>",

            # Legend
            "<s_legend>", "</s_legend>",
            "<s_legend_bbox>", "</s_legend_bbox>",
            "<s_legend_items>", "</s_legend_items>",
            "<s_model>", "</s_model>",
            "<s_label>", "</s_label>",
            "<s_text>", "</s_text>",
            "<s_preview>", "</s_preview>",

            # Models (Series/Plot Data)
            "<s_models>", "</s_models>",
            "<s_name>", "</s_name>",
            "<s_bboxes>", "</s_bboxes>",
            "<s_colors>", "</s_colors>",
            "<s_y>", "</s_y>",
            "<s_labels>", "</s_labels>",
            "<s_x>", "</s_x>",

            # Additional common keys
            "<s_type>", "</s_type>",
            "<s_values>", "</s_values>",
        ]

        model_module.model.decoder.add_special_tokens(common_tokens)
        for split in ["train", "validation"]:
            datasets[split].append(
                DonutDataset(
                    dataset_name_or_path=dataset_name_or_path,
                    donut_model=model_module.model,
                    max_length=config.max_length,
                    split=split,
                    task_start_token=config.task_start_tokens[i]
                    if config.get("task_start_tokens", None)
                    else f"<s_{task_name}>",
                    sort_json_key=config.sort_json_key,
                )
            )
            print(f"DEBUG: Added {split} dataset {task_name}")

    data_module.train_datasets = datasets["train"]
    data_module.val_datasets = datasets["validation"]
    print("DEBUG: Datasets added to data_module")

    logger = TensorBoardLogger(
        save_dir=config.result_path,
        name=config.exp_name,
        version=config.exp_version,
        default_hp_metric=False,
    )
    print("DEBUG: TensorBoard logger initialized")

    lr_callback = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_metric",
        dirpath=Path(config.result_path) / config.exp_name / config.exp_version,
        filename="artifacts",
        save_top_k=1,
        save_last=False,
        mode="min",
    )
    print("DEBUG: Checkpoint callback initialized")

    bar = ProgressBar(config)

    custom_ckpt = CustomCheckpointIO()
    trainer = pl.Trainer(
        num_nodes=config.get("num_nodes", 1),
        devices=torch.cuda.device_count(),
        strategy=config.get("strategy", "auto"),
        accelerator="cuda",
        plugins=custom_ckpt,
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
        val_check_interval=config.val_check_interval,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        gradient_clip_val=config.gradient_clip_val,
        precision=config.get("precision", 16),
        num_sanity_val_steps=config.get("num_sanity_val_steps", 0),
        logger=logger,
        callbacks=[lr_callback, checkpoint_callback, bar],
    )
    print("DEBUG: Trainer initialized")

    trainer.fit(model_module, data_module, ckpt_path=config.get("resume_from_checkpoint_path", None))
    print("DEBUG: Training completed")

import multiprocessing

if __name__ == "__main__":
    print("DEBUG: Starting main block")
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_version", type=str, required=False)
    args, left_argv = parser.parse_known_args()

    config = Config(args.config)
    config.argv_update(left_argv)

    config.exp_name = basename(args.config).split(".")[0]
    config.exp_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if not args.exp_version else args.exp_version

    save_config_file(config, Path(config.result_path) / config.exp_name / config.exp_version)
    print("DEBUG: Configuration file saved")
    
    train(config)
    print("DEBUG: Main block completed")

