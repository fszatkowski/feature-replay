from torch import nn


class Flatten(nn.Module):
    """
    Simple nn.Module to flatten each tensor of a batch of tensors.
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
