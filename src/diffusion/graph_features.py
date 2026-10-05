import torch
import utils


class GraphFeatures:
    """Extra graph features appended to the network input.

    The released checkpoints were all trained with features_type 'none', i.e. no
    extra features beyond X, E and y.
    """

    def __init__(self, extra_features_type, dataset_info):
        self.max_n_nodes = dataset_info.max_n_nodes
        self.features_type = extra_features_type

    def __call__(self, noisy_data):
        if self.features_type != 'none':
            raise ValueError(f"Features type {self.features_type} not implemented")
        X = noisy_data['X_t']
        extra_nodes_attr = torch.zeros((*X.shape[:-1], 0)).type_as(X)
        E = noisy_data['E_t']
        extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
        y = noisy_data['y_t']
        extra_y_attr = torch.zeros((*y.shape[:-1], 0)).type_as(y)
        return utils.PlaceHolder(X=extra_nodes_attr, E=extra_edge_attr, y=extra_y_attr)


# The checkpoints pickle a GraphFeatures instance holding these helper objects.
# They are unused at sampling time, but the names must stay importable so that
# torch.load can restore the checkpoint.
class NodeDegreesFeatures:
    pass


class NodeCycleFeatures:
    pass


class KNodeCycles:
    pass


class EigenFeatures:
    pass
