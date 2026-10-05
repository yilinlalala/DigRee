import torch
import utils


class RNAFeatures:
    def __init__(self, dataset_infos):
        pass

    def __call__(self, noisy_data):
        graph_node_attr = torch.zeros((*noisy_data['X_t'].shape[:-1], 0)).type_as(noisy_data['X_t'])
        graph_edge_attr = torch.zeros((*noisy_data['E_t'].shape[:-1], 0)).type_as(noisy_data['E_t'])
        graph_y_attr = torch.zeros((*noisy_data['y_t'].shape[:-1], 0)).type_as(noisy_data['y_t'])
        return utils.PlaceHolder(X=graph_node_attr, E=graph_edge_attr, y=graph_y_attr)
