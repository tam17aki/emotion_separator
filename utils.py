# -*- coding: utf-8 -*-
"""Utility script.

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
import glob
import os

import numpy as np
import torch
from omegaconf import DictConfig


def init_manual_seed(random_seed: int):
    """Initialize manual seed."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_feats(cfg: DictConfig):
    """Load features and labels.

    Args:
        cfg (DictConfig): configuration of model.

    Returns:
        feats (numpy array) : xvector [3 * 3 * 100, 512]
        labels (numpy array): emotion label [3 * 3 * 100]
    """
    feat_dir = os.path.join(cfg.xvector.root_dir, cfg.xvector.feat_dir)
    feats = {}
    labels = []
    for actor in cfg.actor:
        feats[actor] = []
        for emotion in cfg.emotion:
            feat_files = glob.glob(feat_dir + actor + f"/{actor}_{emotion}_*.npy")
            for feat_file in feat_files:
                xvector = np.load(feat_file)
                xvector = np.expand_dims(xvector, axis=0)
                feats[actor].append(xvector)
                if emotion == "angry":
                    labels.append(0)
                elif emotion == "happy":
                    labels.append(1)
                elif emotion == "normal":
                    labels.append(2)
        feats[actor] = np.concatenate(feats[actor])
    feats = np.concatenate(list(feats.values()))
    labels = np.array(labels)
    return feats, labels


def init_loss_stats():
    """Initialize temporary epoch loss and stats."""
    var = {
        "epoch_loss": 0.0,
        "epoch_loss_enc": 0.0,
        "epoch_loss_adv": 0.0,
        "epoch_loss_cls": 0.0,
        "correct_enc": 0,
        "total_enc": 0,
        "correct_adv": 0,
        "total_adv": 0,
    }
    return var


def update_loss_stats(var, loss, logits, label):
    """Update epoch loss and stats."""
    var["epoch_loss"] += loss["main"].item()
    var["epoch_loss_enc"] += loss["enc"].item()
    var["epoch_loss_adv"] += loss["adv"].item()
    var["epoch_loss_cls"] += loss["cls"].item()
    _, predicted = torch.max(logits["enc"], 1)
    var["correct_enc"] += (predicted == label).sum().item()
    var["total_enc"] += label.size(0)
    _, predicted = torch.max(logits["adv"], 1)
    var["correct_adv"] += (predicted == label).sum().item()
    var["total_adv"] += label.size(0)


def print_loss_acc(var, n_batch: int, is_train: bool):
    """Print loss and accuracy."""
    if is_train:
        print(f" train_loss (main): {var['epoch_loss'] / n_batch:.6f} ", end="")
    else:
        print(
            f"           valid_loss (main): {var['epoch_loss'] / n_batch:.6f} ",
            end="",
        )
    print(f"(enc): {var['epoch_loss_enc'] / n_batch:.6f} ", end="")
    print(f"(cls): {var['epoch_loss_cls'] / n_batch:.6f} ", end="")
    print(f"(adv): {var['epoch_loss_adv'] / n_batch:.6f} - ", end="")
    acc = 100 * float(var["correct_enc"] / var["total_enc"])
    print(
        f"accuracy (enc): {acc:.6f}% ({var['correct_enc']}/{var['total_enc']})",
        end="",
    )
    acc = 100 * float(var["correct_adv"] / var["total_adv"])
    print(f" (adv): {acc:.6f}% ({var['correct_adv']}/{var['total_adv']})")


def save_checkpoint(cfg: DictConfig, modules):
    """Save checkpoint."""
    model = modules.model
    model_dir = os.path.join(cfg.xvector.root_dir, cfg.xvector.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, cfg.training.model_file)
    torch.save(model.state_dict(), model_file)
