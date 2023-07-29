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
import torch


def init_manual_seed(random_seed: int):
    """Initialize manual seed."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
