# -*- coding: utf-8 -*-
"""Training script.

Copyright (C) 2023 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from collections import namedtuple

import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from dataset import get_dataloader
from factory import get_custom_loss, get_lr_scheduler, get_optimizer
from model import get_model
from utils import (init_loss_stats, init_manual_seed, load_feats,
                   print_loss_acc, save_checkpoint, update_loss_stats)


def training_epoch(modules, device):
    """Perform a training epoch."""
    epoch_loss_stats = init_loss_stats()
    modules.model.train()
    parameters = modules.model.grouped_parameters()
    for data, label in modules.dataloader["train"]:
        data = data.to(device).float()
        label = label.to(device).long()
        modules.optimizer["main"].zero_grad()
        modules.optimizer["cls"].zero_grad()

        loss, logits = modules.loss_func(data, label)

        for param in parameters["cls"]:  # freeze AuxiliaryClassifier
            param.requires_grad_(requires_grad=False)
        loss["main"].backward(retain_graph=True)
        modules.optimizer["main"].step()

        for param in parameters["cls"]:  # unfreeze AuxiliaryClassifier
            param.requires_grad_(requires_grad=True)
        for param in parameters["main"]:  # freeze main network
            param.requires_grad_(requires_grad=False)
        loss["cls"].backward()
        modules.optimizer["cls"].step()
        for param in parameters["main"]:  # unfreeze main network
            param.requires_grad_(requires_grad=True)

        update_loss_stats(epoch_loss_stats, loss, logits, label)

    if modules.lr_scheduler is not None:
        modules.lr_scheduler["main"].step()
        modules.lr_scheduler["cls"].step()

    n_batch = len(modules.dataloader["train"])
    print_loss_acc(epoch_loss_stats, n_batch, is_train=True)


def validation_step(modules, device):
    """Perform a validation step."""
    loss_stats = init_loss_stats()
    modules.model.eval()
    with torch.no_grad():
        for data, label in modules.dataloader["test"]:
            data = data.to(device).float()
            label = label.to(device).long()
            loss, logits = modules.loss_func(data, label)
            update_loss_stats(loss_stats, loss, logits, label)
    n_batch = len(modules.dataloader["test"])
    print_loss_acc(loss_stats, n_batch, is_train=False)


def main(cfg: DictConfig):
    """Perform model training."""
    print(OmegaConf.to_yaml(cfg), flush=True)  # dump configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_manual_seed(cfg.training.seed)

    # instantiate modules for training
    feats, labels = load_feats(cfg)
    dataloader = get_dataloader(cfg, feats, labels)
    model = get_model(cfg, device)
    loss_func = get_custom_loss(cfg, model)
    optimizer = get_optimizer(cfg, model)
    lr_scheduler = None
    if cfg.training.use_scheduler:
        lr_scheduler = get_lr_scheduler(cfg, optimizer)

    # pack modules into a namedtuple
    TrainingModules = namedtuple(
        "TrainingModules",
        ["dataloader", "model", "loss_func", "optimizer", "lr_scheduler"],
    )
    modules = TrainingModules(dataloader, model, loss_func, optimizer, lr_scheduler)

    # perform training loop
    for epoch in range(1, cfg.training.n_epoch + 1):
        print(f"Epoch {epoch:2d}: ", end="")
        training_epoch(modules, device)
        if cfg.training.test_size != 0.0:
            validation_step(modules, device)

    # save parameters
    save_checkpoint(cfg, model)


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")
    main(config)
