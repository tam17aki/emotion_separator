# -*- coding: utf-8 -*-
"""Provides customized Dataset.

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

import joblib
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


class XvectorDataset(torch.utils.data.Dataset):
    """Dataset."""

    def __init__(self, feats, labels):
        """Initialize class."""
        self.feats = feats
        self.labels = labels

    def __len__(self):
        """Return the size of the dataset.

        Returns:
            int: size of the dataset
        """
        return self.feats.shape[0]

    def __getitem__(self, idx):
        """Get a pair of input and target.

        Args:
            idx (int): index of the pair

        Returns:
            tuple: input and target in numpy format
        """
        feat = self.feats[idx, :]
        label = self.labels[idx]
        return (feat, label)


def get_dataloader(cfg):
    """Get Dataloaders for training and test.

    Args:
        cfg (DictConfig): configuration of model.

    Returns:
        train_dataloader : dataloader for training
        test_dataloader : dataloader for test
    """
    feats, labels = load_feats(cfg)
    if cfg.training.test_size > 0.0:
        x_train, x_test, y_train, y_test = train_test_split(
            feats,
            labels,
            test_size=cfg.training.test_size,
            random_state=cfg.training.seed,
        )
        scaler = StandardScaler()
        x_train_std = scaler.fit_transform(x_train)
        x_test_std = scaler.transform(x_test)
        train_dataloader = torch.utils.data.DataLoader(
            dataset=XvectorDataset(x_train_std, y_train),
            batch_size=cfg.training.n_batch,
            shuffle=True,
            drop_last=True,
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset=XvectorDataset(x_test_std, y_test),
            batch_size=cfg.training.n_batch,
            shuffle=False,
            drop_last=False,
        )
    else:
        scaler = StandardScaler()
        x_train_std = scaler.fit_transform(feats)
        train_dataloader = torch.utils.data.DataLoader(
            dataset=XvectorDataset(x_train_std, labels),
            batch_size=cfg.training.n_batch,
            shuffle=True,
            drop_last=True,
        )
        test_dataloader = None
    stats_dir = os.path.join(cfg.xvector.root_dir, cfg.xvector.stats_dir)
    os.makedirs(stats_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(stats_dir, cfg.training.scaler_file))
    return {"train": train_dataloader, "test": test_dataloader}
