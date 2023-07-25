# -*- coding: utf-8 -*-
"""Preprocess script.

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
import subprocess

import librosa
import numpy as np
import torch
from hydra import compose, initialize
from progressbar import progressbar as prg
from torchaudio.compliance import kaldi
from xvector_jtubespeech import XVector


def get_corpus(cfg):
    """Download voice-statistics corpurs."""
    corpus_url = cfg.xvector.corpus_url
    data_dir = os.path.join(cfg.xvector.root_dir, cfg.xvector.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    subprocess.run(
        "echo -n Downloading voice statistics corpus ...",
        text=True,
        shell=True,
        check=True,
    )
    for actor in cfg.actor:  # "tsuchiya", "fujitou", "uemura"
        for emotion in cfg.emotion:  # "angry", "happy", "normal"
            command = "wget " + "-P " + "/tmp/" + " " + corpus_url
            tar_file = actor + "_" + emotion + ".tar.gz"
            command = command + tar_file
            subprocess.run(
                command, text=True, shell=True, capture_output=True, check=True
            )
            command = "cd " + data_dir + "; " + "tar -xzvf " + "/tmp/" + tar_file
            subprocess.run(
                command, text=True, shell=True, capture_output=True, check=True
            )
    print(" done.")


def get_pretrained_model(cfg):
    """Download pretrained model."""
    repo_url = cfg.xvector.repo_url
    data_dir = os.path.join(cfg.xvector.root_dir, cfg.xvector.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    model_dir = os.path.join(cfg.xvector.root_dir, cfg.xvector.model_dir)
    os.makedirs(model_dir, exist_ok=True)

    subprocess.run(
        "echo -n Downloading pretrained model ...",
        text=True,
        shell=True,
        check=True,
    )

    # download pretrained model from github repo.b rerained
    command = "wget " + "-P " + "/tmp/" + " " + repo_url
    subprocess.run(command, text=True, shell=True, capture_output=True, check=True)
    command = "cd " + "/tmp/" + "; " + "unzip " + "master.zip"
    subprocess.run(command, text=True, shell=True, capture_output=True, check=True)
    command = (
        "cp "
        + os.path.join("/tmp/", cfg.pretrained.repo_name, cfg.pretrained.file_name)
        + " "
        + os.path.join(model_dir, cfg.pretrained.file_name)
    )
    subprocess.run(command, text=True, shell=True, capture_output=True, check=True)

    # clean up
    command = "rm " + "/tmp/master.zip"
    subprocess.run(command, text=True, shell=True, capture_output=True, check=True)
    command = "rm -rf " + os.path.join("/tmp/", cfg.pretrained.repo_name)
    subprocess.run(command, text=True, shell=True, capture_output=True, check=True)
    print(" done.")


def extract_xvector(cfg):  # xvector model  # 16kHz mono
    "Extract XVectors from wave file."

    feat_dir = os.path.join(cfg.xvector.root_dir, cfg.xvector.feat_dir)
    os.makedirs(feat_dir, exist_ok=True)
    data_dir = os.path.join(cfg.xvector.root_dir, cfg.xvector.data_dir)

    model = XVector(
        os.path.join(
            cfg.xvector.root_dir, cfg.xvector.model_dir, cfg.pretrained.file_name
        )
    )
    for actor in cfg.actor:
        out_dir = os.path.join(feat_dir, actor)
        os.makedirs(out_dir, exist_ok=True)
        for emotion in cfg.emotion:
            data_path = data_dir + actor + "_" + emotion
            wav_list = glob.glob(data_path + f"/{actor}_{emotion}_*.wav")
            for wav_file in prg(
                wav_list, prefix=f"Extract xvectors from {emotion} of {actor}: "
            ):
                wav, _ = librosa.load(wav_file, sr=cfg.feature.sample_rate)
                basename = os.path.basename(wav_file)
                basename, _ = os.path.splitext(basename)
                wav = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
                mfcc = kaldi.mfcc(
                    wav,
                    num_ceps=cfg.feature.num_ceps,
                    num_mel_bins=cfg.feature.num_melbins,
                )
                mfcc = mfcc.unsqueeze(0)
                xvector = model.vectorize(mfcc)
                xvector = xvector.to("cpu").detach().numpy().copy()[0]
                out_path = os.path.join(out_dir, f"{basename}.npy")
                np.save(out_path, xvector)


def main(cfg):
    """Perform preprocess."""
    get_corpus(cfg)
    get_pretrained_model(cfg)
    extract_xvector(cfg)


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")

    main(config)
