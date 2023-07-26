# -*- coding: utf-8 -*-
"""Provides network for splitting emotion components from x-vector.

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
from torch import nn


class SpeakerEncoder(nn.Module):
    """Encoder for speaker component."""

    def __init__(self, cfg: DictConfig):
        """Initialize class."""
        super().__init__()
        input_dim = cfg.model.encoder_spk.input_dim
        hidden_dim = cfg.model.encoder_spk.hidden_dim
        latent_dim = cfg.model.encoder_spk.latent_dim
        n_layers = cfg.model.encoder_spk.n_layers
        layers = nn.ModuleList([])
        for layer in range(n_layers):
            layers += [
                nn.Linear(input_dim if layer == 0 else hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ]
        self.layers = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, inputs):
        """Perform forward propagation.

        Args:
            inputs (Tensor) : x-vector

        Returns:
            logits (Tensor) : logit vector for classification (speaker & emotion)
        """
        hidden = self.layers(inputs)
        outputs = self.fc_out(hidden)
        return outputs


class EmotionEncoder(nn.Module):
    """Encoder for emotion component."""

    def __init__(self, cfg: DictConfig):
        """Initialize class."""
        super().__init__()
        input_dim = cfg.model.encoder_emo.input_dim
        hidden_dim = cfg.model.encoder_emo.hidden_dim
        latent_dim = cfg.model.encoder_emo.latent_dim
        n_layers = cfg.model.encoder_emo.n_layers
        layers = nn.ModuleList([])
        for layer in range(n_layers):
            layers += [
                nn.Linear(input_dim if layer == 0 else hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ]
        self.layers = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, inputs):
        """Perform forward propagation.

        Args:
            inputs (Tensor) : x-vector

        Returns:
            logits (Tensor) : logit vector for classification (emotion)
        """
        hidden = self.layers(inputs)
        logits = self.fc_out(hidden)
        return logits


class Decoder(nn.Module):
    """Decoder for encoded speaker and emotion components."""

    def __init__(self, cfg: DictConfig):
        """Initialize class."""
        super().__init__()
        input_dim = cfg.model.decoder.input_dim
        hidden_dim = cfg.model.decoder.hidden_dim
        latent_dim = cfg.model.encoder_emo.latent_dim + cfg.model.encoder_spk.latent_dim
        n_layers = cfg.model.decoder.n_layers
        layers = nn.ModuleList([])
        for layer in range(n_layers):
            layers += [
                nn.Linear(latent_dim if layer == 0 else hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ]
        self.layers = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, inputs):
        """Perform forward decoder propagation.

        Args:
            inputs (Tensor) : latent vector

        Returns:
            outputs (Tensor) : reconstructed x-vector
        """
        hidden = self.layers(inputs)
        outputs = self.fc_out(hidden)
        return outputs


class AuxiliaryClassifier(nn.Module):
    """Auxiliary classifier for speaker recog and emotion recog."""

    def __init__(self, cfg: DictConfig, n_classes=3):
        """Initialize class."""
        super().__init__()
        input_dim = cfg.model.encoder_spk.latent_dim
        hidden_dim = cfg.model.classifier_aux.hidden_dim
        n_layers = cfg.model.classifier_aux.n_layers
        layers = nn.ModuleList([])
        for layer in range(n_layers):
            layers += [
                nn.Linear(input_dim if layer == 0 else hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ]
        self.layers = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dim, n_classes)

    def forward(self, inputs):
        """Perform forward propagation.

        Args:
            inputs (Tensor) : latent vector (speaker)

        Returns:
            logits (Tensor) : logit vector for classification
        """
        hidden = self.layers(inputs)
        logits = self.fc_out(hidden)
        return logits


class EmotionComponentSplitter(nn.Module):
    """Class EmotionComponentSplitter.

    This class implements a functionality to split emotion components from x-vector.
    """

    def __init__(self, cfg: DictConfig):
        """Initialize class.

        Args:
            cfg (DictConfig): configuration of model.
        """
        super().__init__()
        self.speaker_encoder = SpeakerEncoder(cfg)
        self.emotion_encoder = EmotionEncoder(cfg)
        self.emotion_projection = nn.Linear(
            cfg.model.encoder_emo.latent_dim, len(cfg.emotion)
        )
        self.auxiliary_classifier = AuxiliaryClassifier(cfg)
        self.decoder = Decoder(cfg)
        self.criterion = {"ce": nn.CrossEntropyLoss(), "mse": nn.MSELoss()}
        self.cfg = cfg

    def grouped_parameters(self):
        """Perform forward propagation.

        Returns:
            parameters (dict): model parameters of aux classifier and others.
        """
        params_group = list(self.speaker_encoder.parameters())
        params_group.extend(list(self.emotion_encoder.parameters()))
        params_group.extend(list(self.emotion_projection.parameters()))
        params_group.extend(list(self.decoder.parameters()))
        parameters = {
            "main": params_group,
            "cls": list(self.auxiliary_classifier.parameters()),
        }
        return parameters

    def forward(self, inputs):
        """Perform forward propagation."""
        spk_latent = self.speaker_encoder(inputs)
        emo_latent = self.emotion_encoder(inputs)
        logits_enc = self.emotion_projection(emo_latent)
        logits_aux = self.auxiliary_classifier(spk_latent)
        latent = torch.cat((spk_latent, emo_latent), dim=1)
        reconst = self.decoder(latent)
        return reconst, logits_enc, logits_aux

    def get_speaker_latent(self, inputs):
        """Returns latent vector of speaker component."""
        spk_latent = self.speaker_encoder(inputs)
        return spk_latent


def get_model(cfg: DictConfig, device: torch.device):
    """Instantiate network."""
    return EmotionComponentSplitter(cfg).to(device)
