# -*- coding: utf-8 -*-
"""Inference script to convert original x-vector into emotion-separated one.

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
from hydra import compose, initialize
from omegaconf import DictConfig

from model import get_model
from utils import load_checkpoint


def main(cfg: DictConfig):
    """Perform inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg, device)
    load_checkpoint(cfg, model)
    model.eval()
    feat_dir = os.path.join(cfg.xvector.root_dir, cfg.xvector.feat_dir)
    out_dir = os.path.join(cfg.xvector.root_dir, cfg.xvector.sep_dir)
    os.makedirs(out_dir, exist_ok=True)
    for actor in cfg.actor:
        for emotion in cfg.emotion:
            feat_files = glob.glob(feat_dir + actor + f"/{actor}_{emotion}_*.npy")
            for feat_file in feat_files:
                xvector = np.load(feat_file)
                xvector = np.expand_dims(xvector, axis=0)
                xvector = torch.from_numpy(xvector.astype(np.float32))
                xvector = xvector.to(device)
                latent = model.get_speaker_latent(xvector)
                latent = latent.to("cpu").detach().numpy().copy()
                latent = latent.squeeze(0)
                basename = os.path.basename(feat_file)
                basename, _ = os.path.splitext(basename)
                out_path = os.path.join(out_dir, f"{basename}.npy")
                np.save(out_path, latent)


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")
    main(config)
