# -*- coding: utf-8 -*-
"""A script for visualization of xvectors via UMAP.

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

import matplotlib.pyplot as plt
import numpy as np
from hydra import compose, initialize
from umap import UMAP


def get_emotion_type(actor, emotion):
    """Return emotion type."""
    emotion_type = -1
    if emotion == "angry" and actor == "tsuchiya":
        emotion_type = 0
    elif emotion == "happy" and actor == "tsuchiya":
        emotion_type = 1
    elif emotion == "normal" and actor == "tsuchiya":
        emotion_type = 2
    elif emotion == "angry" and actor == "fujitou":
        emotion_type = 3
    elif emotion == "happy" and actor == "fujitou":
        emotion_type = 4
    elif emotion == "normal" and actor == "fujitou":
        emotion_type = 5
    elif emotion == "angry" and actor == "uemura":
        emotion_type = 6
    elif emotion == "happy" and actor == "uemura":
        emotion_type = 7
    elif emotion == "normal" and actor == "uemura":
        emotion_type = 8

    return emotion_type


def main(cfg):
    """Perform UMAP plot in 2D space."""
    feat_dir = os.path.join(cfg.xvector.root_dir, cfg.xvector.sep_dir)
    feat_list = []
    emotion_type = []
    for actor in cfg.actor:
        for emotion in cfg.emotion:
            feats_files = glob.glob(feat_dir + "/" + f"{actor}_{emotion}_*.npy")
            for feats in feats_files:
                xvector = np.load(feats)
                xvector = np.expand_dims(xvector, axis=0)
                feat_list.append(xvector)
                emotion_type.append(get_emotion_type(actor, emotion))

    feat_array = np.concatenate(feat_list)
    mapper = UMAP(n_components=2, random_state=cfg.inference.seed)
    fit = mapper.fit(feat_array)
    embedding = fit.transform(feat_array)
    embedding_x = embedding[:, 0]
    embedding_y = embedding[:, 1]
    for n in np.unique(emotion_type):
        if n == 0:
            label = "tsuchiya-angry"
        elif n == 1:
            label = "tsuchiya-happy"
        elif n == 2:
            label = "tsuchiya-normal"
        elif n == 3:
            label = "fujitou-angry"
        elif n == 4:
            label = "fujitou-happy"
        elif n == 5:
            label = "fujitou-normal"
        elif n == 6:
            label = "uemura-angry"
        elif n == 7:
            label = "uemura-happy"
        elif n == 8:
            label = "uemura-normal"
        plt.scatter(
            embedding_x[emotion_type == n],
            embedding_y[emotion_type == n],
            label=label,
            s=10,
        )
    plt.grid()
    plt.legend()
    plt.tight_layout()
    img_dir = os.path.join(cfg.xvector.root_dir, cfg.xvector.img_dir)
    os.makedirs(img_dir, exist_ok=True)
    img_file = os.path.join(
        img_dir,
        cfg.inference.umap_image_file
        + f"_epoch{cfg.training.n_epoch}"
        + f"_adv{cfg.training.adv_weight}"
        + f"_cls{cfg.training.cls_weight}"
        + cfg.inference.umap_image_ext,
    )
    plt.savefig(img_file)
    plt.show()


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")

    main(config)
