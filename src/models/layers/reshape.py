from torch import nn


class Reshape(nn.Module):
    """
    Simple nn.Module to flatten each tensor of a batch of tensors.
    """

    def __init__(self,
                 shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, *self.shape)
