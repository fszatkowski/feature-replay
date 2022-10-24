import numpy as np
from avalanche.training.plugins import SupervisedPlugin


class RandomPartialFreezingPlugin(SupervisedPlugin):
    def __init__(self, probs: list[float]):
        """
        Random Partial Freezing plugin. For each batch, randomly disables gradient update in parts
        of the network.
        :param probs: List of probabilities for disabling layers. Should be equal to the number of
        layers in the model and sum to 1.0. List index indicates the layer at which the gradient
        update will be cut, eg: with list [0.5, 0.3, 0.2] we have 0.5 chance that the gradient will
        be cut at layer 0 (meaning backpropagation through the whole network), 0.3 chance that
        gradient will not be back-propagated through the first layer and 0.2 chance that gradient
        will not be back-propagated through the first two layers.
        """
        super().__init__()
        assert sum(probs) == 1.0, (
            f"`probs` passed to `RandomPartialFreezingPlugin` should sum "
            f"to one, but provided `probs`={probs} sum to {sum(probs)}."
        )
        self.probs = probs

    def before_training(self, strategy, *args, **kwargs):
        assert len(self.probs) == strategy.model.n_layers(), (
            f"Length of the `probs` list for `RandomPartialFreezingPlugin` should be equal to "
            f"{strategy.model.n_layers()}, but got `probs`={self.probs}"
        )

    def before_update(self, strategy, *args, **kwargs) -> None:
        freeze_thresh = np.random.choice(
            list(range(len(self.probs))), 1, p=self.probs
        ).item()
        for layer in strategy.model.layers[:freeze_thresh]:
            layer.requires_grad_(False)

    def after_update(self, strategy, *args, **kwargs) -> None:
        for layer in strategy.model.layers:
            layer.requires_grad_(True)
