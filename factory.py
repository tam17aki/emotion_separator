# -*- coding: utf-8 -*-
"""A Python module which provides optimizer, scheduler, and customized loss.

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
import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.nn.functional import softmax


def get_optimizer(cfg: DictConfig, model):
    """Instantiate optimizer."""
    parameters = model.grouped_parameters()
    optimizer_class = getattr(optim, cfg.training.optim.optimizer.name)
    optimizer_main = optimizer_class(
        parameters["main"], **cfg.training.optim.optimizer.params
    )
    optimizer_cls = optimizer_class(
        parameters["cls"], **cfg.training.optim.optimizer.params
    )
    return {"main": optimizer_main, "cls": optimizer_cls}


def get_lr_scheduler(cfg: DictConfig, optimizers):
    """Instantiate scheduler."""
    lr_scheduler_class = getattr(
        optim.lr_scheduler, cfg.training.optim.lr_scheduler.name
    )
    lr_scheduler_main = lr_scheduler_class(
        optimizers["main"], **cfg.training.optim.lr_scheduler.params
    )
    lr_scheduler_cls = lr_scheduler_class(
        optimizers["cls"], **cfg.training.optim.lr_scheduler.params
    )
    return {"main": lr_scheduler_main, "cls": lr_scheduler_cls}


class CustomLoss(nn.Module):
    """Custom loss."""

    def __init__(self, cfg: DictConfig, model):
        """Initialize class."""
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.criterion = {"ce": nn.CrossEntropyLoss(), "mse": nn.MSELoss()}

    def forward(self, inputs, labels):
        """Compute loss function.

        Args:
            inputs (Tensor) : xvector [batch_size, 512]
            labels (Tensor): emotion label [batch_size]
            model (nn.Module): network

        Returns:
            loss_main: main loss
                       - reconstruction loss (MSE)
                       - encoder loss (CE)
                       - adversarial loss (CE or MSE)
            loss_enc:  emotion 'classification' loss
            loss_adv:  emotion 'adversarial' loss
            loss_cls:  emotion 'classification' loss
        """
        reconst, logits_enc, logits_aux = self.model(inputs)

        # compute reconstruction loss
        loss_rec = self.criterion["mse"](inputs, reconst)

        # compute emotion 'encoder' loss with projection matrix with pro
        loss_enc = self.criterion["ce"](logits_enc, labels)

        # compute emotion auxiliary 'classification' loss
        loss_cls = self.criterion["ce"](logits_aux, labels)

        # compute emotion 'adversarial' loss
        if self.cfg.training.ce_loss_adv:
            loss_adv = -loss_cls  # negative cross entropy loss for adversarial loss
        else:
            uniform = 1 / logits_aux.size(1) * torch.ones_like(logits_aux)
            # uniform distribution of emotion class
            loss_adv = self.criterion["mse"](softmax(logits_aux, dim=1), uniform)

        loss_dict = {
            "main": loss_rec + loss_enc + self.cfg.training.adv_weight * loss_adv,
            "enc": loss_enc,
            "adv": loss_adv,
            "cls": loss_cls,
        }
        logits_dict = {"enc": logits_enc, "adv": logits_aux}
        return loss_dict, logits_dict


def get_loss(cfg: DictConfig, model):
    """Instantiate customized loss."""
    custom_loss = CustomLoss(cfg, model)
    return custom_loss
