import random

import torch


class DriftAnalysisBuffer:
    def __init__(
        self,
        memory_size: int,
        expanding_buffer: bool,
        n_experiences: int,
        n_classes: int,
        n_layers: int,
    ):
        """
        Buffer containing features for drift analysis. After each experience, adds original images
        along with representations at each level this experience and balances the buffer so that
        each experience is represented by roughly the same number of examples (can vary if ratio
        of memory size to number of experiences is not integer).
        :param memory_size: Memory size for buffer.
        :param expanding_buffer: Whether the buffer should expand. If True, after each experience,
        the `memory_size` examples will be added to the buffer. If False, examples from the new
        experience will be added so that the buffer is balanced, but it's size stays constant.
        :param n_experiences: Total number of experiences.
        """
        self.memory_size = memory_size
        self.expanding_buffer = expanding_buffer
        self.n_experiences = n_experiences
        self.n_classes = n_classes

        self.buffer: dict[int, list] = {i: [] for i in range(n_experiences)}
        self.init_features: dict[int, list[dict]] = {
            i: [] for i in range(n_experiences)
        }
        self.per_exp_drift: dict[int, dict[int, list]] = {
            i: {j: [] for j in range(1, n_layers)} for i in range(n_experiences)
        }

    def after_training_epoch(self, strategy) -> None:
        for exp_id in range(self.n_experiences):
            if len(self.buffer[exp_id]) > 0:
                loss = torch.nn.CosineEmbeddingLoss()
                model_n_layers = strategy.model.n_layers()
                per_layer_cos_dists: dict[int, list] = {
                    i: [] for i in range(1, model_n_layers)
                }

                for (feature, label), base_features in zip(
                    self.buffer[exp_id], self.init_features[exp_id]
                ):
                    layer_id = 0
                    feature = feature.unsqueeze(0)
                    while layer_id < model_n_layers - 1:
                        feature = strategy.model(
                            feature.to("cuda"),
                            skip_first=layer_id,
                            skip_last=model_n_layers - layer_id - 1,
                        )
                        base_feature = base_features[layer_id + 1]
                        per_layer_cos_dists[layer_id + 1].append(
                            loss(
                                feature.to("cpu"),
                                base_feature.to("cpu"),
                                target=torch.ones(1),
                            ).detach()
                        )
                        layer_id += 1

                for layer_idx, cos_dists in per_layer_cos_dists.items():
                    mean_dist = torch.mean(torch.stack(cos_dists)).item()
                    self.per_exp_drift[exp_id][layer_idx].append(mean_dist)
            else:
                for drift in self.per_exp_drift[exp_id].values():
                    drift.append(0.0)

    def after_training_exp(self, strategy) -> None:
        """
        Update the buffer on the last experience.
        """
        dataset = strategy.experience.dataset
        exp_id = strategy.clock.train_exp_counter
        total_n_exp = self.n_experiences
        classes_per_exp = self.n_classes / total_n_exp

        if self.expanding_buffer:
            label_to_features = {}
            memory_per_class = self.memory_size / classes_per_exp
            dataset_idx = list(range(len(dataset)))
            random.shuffle(dataset_idx)
            for idx in dataset_idx:
                x, y, e = dataset[idx]
                if y not in label_to_features.keys():
                    label_to_features[y] = [x]
                elif len(label_to_features[y]) < memory_per_class:
                    label_to_features[y].append(x)
                elif len(label_to_features) == classes_per_exp and all(
                    len(features) == memory_per_class
                    for label, features in label_to_features.items()
                ):
                    break
            buffer = [
                (feature.to("cpu"), label)
                for label, per_label_features in label_to_features.items()
                for feature in per_label_features
            ]
        else:
            raise NotImplementedError()

        self.buffer[exp_id] = buffer

        model_n_layers = strategy.model.n_layers()
        for feature, label in buffer:
            layer_id = 0
            features_dict = {}
            feature = feature.unsqueeze(0)
            while layer_id < model_n_layers - 1:
                feature = strategy.model(
                    feature.to("cuda"),
                    skip_first=layer_id,
                    skip_last=model_n_layers - layer_id - 1,
                )
                features_dict[layer_id + 1] = feature.to("cpu")
                layer_id += 1
            self.init_features[exp_id].append(features_dict)
