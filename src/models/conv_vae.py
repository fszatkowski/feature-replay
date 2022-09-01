################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 03-03-2022                                                             #
# Author: Florian Mies                                                         #
# Website: https://github.com/travela                                          #
################################################################################

"""

File to place any kind of generative models
and their respective helper functions.

"""
from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from avalanche.models.base_model import BaseModel
from torch import Tensor
from torchvision import transforms

from models.layers.dense import DenseLayer
from models.layers.flatten import Flatten
from models.layers.conv import ConvLayer
from models.layers.conv_transpose import ConvTransposeLayer
from models.layers.reshape import Reshape


class Generator(BaseModel):
    """
    A base abstract class for generators
    """

    @abstractmethod
    def generate(self, batch_size=None, condition=None):
        """
        Lets the generator sample random samples.
        Output is either a single sample or, if provided,
        a batch of samples of size "batch_size"

        :param batch_size: Number of samples to generate
        :param condition: Possible condition for a condotional generator
                          (e.g. a class label)
        """


###########################
# VARIATIONAL AUTOENCODER #
###########################


class VAEConvEncoder(nn.Module):
    """
    Encoder part of the VAE, computer the latent represenations of the input.

    :param shape: Shape of the input to the network: (channels, height, width)
    :param hidden_sizes: Hidden layer sizes
    """

    def __init__(
            self,
            input_size: [list[int]] = (1, 28, 28),
            channels: Optional[list[int]] = None,
            kernel_size: int = 4,
            pooling: bool = False,
            hidden_sizes: Optional[list[int]] = None
    ) -> object:
        super(VAEConvEncoder, self).__init__()
        self.layers = nn.Sequential()

        if hidden_sizes is None:
            hidden_sizes = [512]
        if channels is None:
            channels = [64]

        for layer_idx, (in_channels, out_channels) in enumerate(
                zip([input_size[0]] + channels, channels)
        ):
            self.layers.add_module(
                f"conv{layer_idx}",
                ConvLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    pooling=pooling,
                    dropout_ratio=0,
                    flatten=False,
                ),
            )

        hidden_input_size = input_size[1] * input_size[2] * channels[-1]
        if pooling:
            hidden_input_size = int(hidden_input_size / 4 ** len(channels))

        self.layers.add_module("input_flattened", Flatten())

        for layer_idx, (in_size, out_size) in enumerate(
                zip([hidden_input_size] + hidden_sizes[:-1], hidden_sizes)
        ):
            self.layers.add_module(
                f"fc{layer_idx}",
                DenseLayer(
                    in_size=in_size,
                    out_size=out_size,
                    activation=True,
                    dropout_ratio=0,
                ),
            )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class VAEConvDecoder(nn.Module):
    """
    Decoder part of the VAE. Reverses Encoder.

    :param shape: Shape of output: (channels, height, width).
    :param nhid: Dimension of input.
    """

    def __init__(
            self,
            input_size: [list[int]] = (1, 28, 28),
            channels: Optional[list[int]] = None,
            kernel_size: int = 4,
            strides: [list[int]] = None,
            hidden_sizes: Optional[list[int]] = None,
            nhid: int = 2,
    ):

        super(VAEConvDecoder, self).__init__()

        self.input_size = input_size
        self.layers = nn.Sequential()

        if hidden_sizes is None:
            hidden_sizes = [512]
        if channels is None:
            channels = [64]
        if strides is None:
            strides = [2]

        size_before_conv_transpose = [int(channels[-1]),
                                      int(input_size[1] / 2 ** len(channels)),
                                      int(input_size[1] / 2 ** len(channels))]
        flattened_size_before_conv_transpose = torch.Size(size_before_conv_transpose).numel()

        for layer_idx, (in_size, out_size) in enumerate(
                zip([nhid] + hidden_sizes, hidden_sizes + [flattened_size_before_conv_transpose])
        ):
            self.layers.add_module(
                f"fc{layer_idx}",
                DenseLayer(
                    in_size=in_size,
                    out_size=out_size,
                    activation=True,
                    dropout_ratio=0,
                ),
            )

        self.layers.add_module("reshape", Reshape(size_before_conv_transpose))

        for layer_idx, (in_channels, out_channels, stride) in enumerate(
                zip(channels, channels[:-1] + [input_size[0]], strides)
        ):
            self.layers.add_module(
                f"conv{layer_idx}",
                ConvTransposeLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    flatten=False,
                    activation=True if layer_idx != (len(channels) - 1) else False,
                ),
            )

        self.last_layer = nn.Sigmoid()

        # TODO: Check if generetive replay from avalanche
        # was hardcoded to only work for MNISt?
        # If yes - make it a general implementation
        self.invTrans = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])

    def forward(self, z: Tensor) -> Tensor:
        for layer in self.layers:
            z = layer(z)
        z = self.last_layer(z)
        return self.invTrans(z.view(-1, *self.input_size))


class ConvVAE(Generator, nn.Module):
    """
    Variational autoencoder module:
    fully-connected and suited for any input shape and type.

    The encoder only computes the latent represenations
    and we have then two possible output heads:
    One for the usual output distribution and one for classification.
    The latter is an extension the conventional VAE and incorporates
    a classifier into the network.
    More details can be found in: https://arxiv.org/abs/1809.10635
    """

    def __init__(
            self,
            shape: [list[int]] = (1, 28, 28),
            nhid: int = 2,
            channels: Optional[list[int]] = None,
            kernel_size: int = 4,
            strides: [list[int]] = None,
            hidden_sizes: Optional[list[int]] = None,
            device="cpu",
    ):
        """
        :param shape: Shape of each input sample
        :param nhid: Dimension of latent space of Encoder.
        """
        super(ConvVAE, self).__init__()
        self.dim = nhid
        self.device = device
        self.encoder = VAEConvEncoder(input_size=shape,
                                      kernel_size=kernel_size,
                                      channels=channels,
                                      pooling=True,
                                      hidden_sizes=hidden_sizes)
        self.calc_mean = nn.Linear(hidden_sizes[-1], nhid)
        self.calc_logvar = nn.Linear(hidden_sizes[0], nhid)
        self.decoder = VAEConvDecoder(input_size=shape,
                                      kernel_size=kernel_size,
                                      channels=channels[::-1],
                                      strides=strides,
                                      nhid=nhid,
                                      hidden_sizes=hidden_sizes[::-1])

    def get_features(self, x):
        """
        Get features for encoder part given input x
        """
        return self.encoder(x)

    def generate(self, batch_size=None):
        """
        Generate random samples.
        Output is either a single sample if batch_size=None,
        else it is a batch of samples of size "batch_size".
        """
        z = (
            torch.randn((batch_size, self.dim)).to(self.device)
            if batch_size
            else torch.randn((1, self.dim)).to(self.device)
        )
        res = self.decoder(z)
        if not batch_size:
            res = res.squeeze(0)
        return res

    def sampling(self, mean, logvar):
        """
        VAE 'reparametrization trick'
        """
        eps = torch.randn(mean.shape).to(self.device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x):
        """
        Forward.
        """
        represntations = self.encoder(x)
        mean, logvar = self.calc_mean(represntations), self.calc_logvar(represntations)
        z = self.sampling(mean, logvar)
        return self.decoder(z), mean, logvar


# Loss functions
BCE_loss = nn.BCELoss(reduction="sum")
MSE_loss = nn.MSELoss(reduction="sum")
CE_loss = nn.CrossEntropyLoss()


def VAE_loss(X, forward_output):
    """
    Loss function of a VAE using mean squared error for reconstruction loss.
    This is the criterion for VAE training loop.

    :param X: Original input batch.
    :param forward_output: Return value of a VAE.forward() call.
                Triplet consisting of (X_hat, mean. logvar), ie.
                (Reconstructed input after subsequent Encoder and Decoder,
                mean of the VAE output distribution,
                logvar of the VAE output distribution)
    """
    X_hat, mean, logvar = forward_output
    reconstruction_loss = MSE_loss(X_hat, X)
    KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean ** 2)
    return reconstruction_loss + KL_divergence


__all__ = ["MlpVAE", "VAE_loss"]
