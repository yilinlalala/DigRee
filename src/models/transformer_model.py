import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor
import utils
from diffusion import diffusion_utils
from models.layers import Xtoy, Etoy, masked_softmax
from scipy.sparse.csgraph import floyd_warshall
from collections import Counter, defaultdict
from torch_geometric.nn import MessagePassing

class XEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int =2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()


        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw)
        
        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, X: Tensor, E: Tensor, y, node_mask: Tensor):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """

        # Graph Transfomer 
        newX, newE, new_y= self.self_attn(X, E, y, node_mask=node_mask)
        # print("attenscore",attenscore.shape)
        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)
        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)
        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)
        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)
        # print(y.shape)
        return X, E, y
class NodeEdgeBlock(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """
    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        self.y_e_mul = Linear(dy, dx)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)
        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)


        # # Process y
        self.y_y = Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))


    def forward(self, X, E, y, node_mask):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        isnan_X = torch.isnan(X).any().item()
        assert isnan_X == False, "isnan_X:True"

        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask           # (bs, n, dx)
        K = self.k(X) * x_mask           # (bs, n, dx)
        nan_mask = torch.isnan(Q)
        Q[nan_mask] = 0
        diffusion_utils.assert_correctly_masked(Q, x_mask)

        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df
        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))
        Q = Q.unsqueeze(2)                              # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)                              # (bs, n, 1, n head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        nan_mask = torch.isnan(Y)
        Y[nan_mask] = 0
        diffusion_utils.assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

        E1 = self.e_mul(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))
        E2 = self.e_add(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2                 # (bs, n, n, n_head, df)

        # Incorporate y to E
        newE = Y.flatten(start_dim=3)                      # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)    # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2   # bs, n, n, de

        nan_mask = torch.isnan(newE)
        newE[nan_mask] = 0
        diffusion_utils.assert_correctly_masked(newE, e_mask1 * e_mask2)

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)    # bs, 1, n, 1
        attn = masked_softmax(Y, softmax_mask, dim=2)            # bs, n, n, n_head

        V = self.v(X) * x_mask                                   # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)                                       # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)
        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, dx

        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = weighted_V
        newX = yx1 + (yx2 + 1) * weighted_V
        newX = self.x_out(newX) * x_mask
        nan_mask = torch.isnan(newX)
        newX[nan_mask] = 0
        diffusion_utils.assert_correctly_masked(newX, x_mask)

        # Process y based on X axnd E
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        # new_y = y
        new_y = self.y_out(new_y)               # bs, dy
        return newX, newE, new_y
    
class GraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']

        self.mlp_in_X = nn.Sequential(nn.Linear(5*input_dims['X'], hidden_mlp_dims['X']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['E'], hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)
        
        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'])
                                        for i in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims['X']))

        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims['E']))

        self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['y'], output_dims['y']))

        self.sublist_processor = SublistAndIndexCalculator()
        self.directional_mp = CustomMessagePassing()
        self.conv_net = ConvNet(hidden_dims['dx'],hidden_dims['dx'])

        self.pos_x_emb = PositionalXEmbedding(hidden_dims['pos_x'])
        self.noise_emb = TimestepEmbedding(hidden_dims['pos_x'])
        self.sp_emb = SPEmbedding(hidden_dims['pos_x'])
        self.kmer_emb = KMerEmbedding(2*hidden_dims['pos_x'])

    def forward(self, X, E, y, node_mask,t):

        bs, n = X.shape[0], X.shape[1]
        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)
        diag_maskX =  torch.ones(n)
        diag_maskX = ~diag_maskX.type_as(X).bool()
        diag_maskX = diag_maskX.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1)

        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y]

        X_pos_emb = self.pos_x_emb(X)
        noise_emb = self.noise_emb(t,X)
        sp_emb = self.sp_emb(E) * diag_maskX
        Kmer_emb = self.kmer_emb(X) * diag_maskX

        X_pos_kmer = torch.cat((X_pos_emb, Kmer_emb), dim=-1)
        X_pos_kmer_sp = torch.cat((X_pos_kmer, sp_emb), dim=-1)
        X_pos_kmer_sp_nl = torch.cat((X_pos_kmer_sp, noise_emb), dim=-1)

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2

        after_in = utils.PlaceHolder(X=self.mlp_in_X(X_pos_kmer_sp_nl), E=new_E, y=self.mlp_in_y(y)).mask(node_mask)
        
        X, E, y = after_in.X, after_in.E, after_in.y

        for i, layer in enumerate(self.tf_layers[:-1]):
            X, E, y = layer(X, E, y, node_mask)

        edge_index = self.sublist_processor.extract_edge_index(E) # train cost 0.2968759536743164, 16*2*list
        X_DirectionalMP = torch.zeros_like(X)
        for b in range(bs):
            edge_index_b = edge_index[b].to(X.device)
            if edge_index_b.dim() == 0:
                edge_index_b = torch.empty((2, 0), dtype=torch.long)
            self.sublist_processor.process_and_calculate(edge_index_b)
            result = self.sublist_processor.get_indices() # 1.3813259601593018 -> train cost 0.021357059478759766
            if not result['loop_indices']:
                X_DirectionalMP[b] = X[b]
            else:
                X_DirectionalMP[b] = self.directional_mp(X[b], result['loop_indices'], result['stem_indices'])

        X_processed = self.conv_net(X_DirectionalMP)
        X, E, y= self.tf_layers[-1](X_processed, E, y, node_mask)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)
        # print("mlp_out",y.shape)
        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out
        E = 1/2 * (E + torch.transpose(E, 1, 2))
        return utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)

class SPEmbedding(nn.Module):
    def __init__(self, out_dim):
        super(SPEmbedding, self).__init__()
        self.out_dim = out_dim
        self.max_dim = 201
        self.linear = nn.Linear(self.max_dim, out_dim)
    def spatial_pos_process(self, E):
        adjacency_matrix = (torch.argmax(E, dim=-1) != 0).float()
        shortest_path_results = []
        for adjacency_matrix_single in adjacency_matrix:
            shortest_path_result_single = floyd_warshall(adjacency_matrix_single.cpu().numpy())
            shortest_path_results.append(shortest_path_result_single[..., np.newaxis])
        spatial_pos = torch.from_numpy(np.array(shortest_path_results)).long()
        spatial_pos[spatial_pos < 0] = 0
        return spatial_pos
    def forward(self, E):
        spatial_pos = self.spatial_pos_process(E).squeeze(dim=-1).to(E.device)
        padded_spatial_pos = torch.zeros((spatial_pos.size(0), self.max_dim, self.max_dim), dtype=torch.long).to(E.device)
        padded_spatial_pos[:, :spatial_pos.size(1), :spatial_pos.size(2)] = spatial_pos
        emb_spatial_max = self.linear(padded_spatial_pos.float())
        emb_spatial = emb_spatial_max[:, :spatial_pos.size(1), :]
        return emb_spatial

class KMerEmbedding(nn.Module):
    def __init__(self, embed_dim,input_dim=320):
        super(KMerEmbedding, self).__init__()
        self.kmerlinearprocess = Kmerlinearprocess()
        self.embedding = nn.Linear(input_dim, embed_dim)

    def forward(self, X):
        change2seq = ChangeXtoseq()
        seqs = change2seq.to_augc_sequence(X)
        kmer_embeddings = []
        for seq in seqs:
            kmer_probabilities = self.kmerlinearprocess.calculate_kmer_probabilities(seq)
            kmer_probabilities_tensor = torch.tensor(kmer_probabilities, dtype=torch.float32).to(X.device)
            embedded_kmer_probabilities = self.embedding(kmer_probabilities_tensor)
            kmer_embeddings_np = embedded_kmer_probabilities.squeeze(0).cpu().detach().numpy()
            kmer_embeddings.append(kmer_embeddings_np)
        kmer_embeddings_np = np.array(kmer_embeddings, dtype=np.float32)
        embedded_kmers = torch.tensor(kmer_embeddings_np, dtype=torch.float32).to(X.device)
        n = X.shape[1]  
        embedded_kmers = embedded_kmers.unsqueeze(1).expand(-1, n, -1)
        return embedded_kmers

class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_positions=1000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_positions = max_positions
        self.half_dim = embedding_dim // 2
        self.emb = self._create_embedding()
    def _create_embedding(self):
        emb = math.log(self.max_positions) / (self.half_dim - 1)
        emb = torch.exp(torch.arange(self.half_dim, dtype=torch.float32) * -emb)
        return emb
    def forward(self, timesteps,X):
        num_nodes = X.shape[1]
        assert len(timesteps.shape) == 2 and timesteps.shape[1] == 1
        batch_size = timesteps.shape[0]
        emb = timesteps.float() * self.emb[None, :].to(X.device)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode='constant')
        emb = emb.unsqueeze(1).expand(batch_size, num_nodes, self.embedding_dim)
        return emb

class Kmerlinearprocess:
    def __init__(self, k_min=3, k_max=4):
        self.k_min = k_min
        self.k_max = k_max
        self.nucleotide_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3}

    def nucleotide_to_index(self, nucleotide):
        return self.nucleotide_map.get(nucleotide, -1)
    
    def kmer_to_index(self, kmer):
        base = 4
        index = 0
        for nucleotide in kmer:
            index = index * base + int(nucleotide)
        return index

    def calculate_kmer_probabilities(self, sequence):
        sequence = [self.nucleotide_to_index(n) for n in sequence]
        kmer_probabilities = []
        for k in range(self.k_min, self.k_max + 1):
            kmer_counts = Counter([tuple(sequence[i:i+k]) for i in range(len(sequence) - k + 1)])
            total_kmers = len(sequence) - k + 1
            kmer_prob = {kmer: count / total_kmers for kmer, count in kmer_counts.items()}
            kmer_vector = np.zeros(4**k)
            for kmer, prob in kmer_prob.items():
                index = self.kmer_to_index(kmer)
                kmer_vector[index] = prob
            kmer_probabilities.append(kmer_vector)
        return np.concatenate(kmer_probabilities)

class PositionalXEmbedding(nn.Module):
    def __init__(self, num_hiddens, max_len=1000):
        super(PositionalXEmbedding, self).__init__()
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)  # [1, 1000, 4]
    def forward(self, X):
        pos_x_emb = X + self.P[:, :X.shape[1], :X.shape[2]].to(X.device)
        return pos_x_emb

class ChangeXtoseq:
    def __init__(self):
        self.mapping = {0: 'A', 1: 'U', 2: 'G', 3: 'C'}
    def to_augc_sequence(self, X):
        augc_sequences = []
        for seq in X:
            augc_seq = ''.join([self.mapping[idx.item()] for idx in torch.argmax(seq, dim=-1)])
            augc_sequences.append(augc_seq)
        return augc_sequences

class ConvNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, X):
        X = X.transpose(1, 2)  
        X = self.conv1(X)
        X = self.relu(X)
        X = X.transpose(1, 2)  
        return X

class CustomMessagePassing(MessagePassing):
    def __init__(self, aggr='add'):
        super(CustomMessagePassing, self).__init__(aggr=aggr)
        self.alpha = 0.5

    def forward(self, x, loop_indices, stem_indices):
        stem_edge_index = self.get_edge_index_from_indices_interconnect(stem_indices).to(x.device)
        loop_edge_index = self.get_edge_index_from_indices_interconnect(loop_indices).to(x.device)
        combined_edge_index = torch.cat([stem_edge_index, loop_edge_index], dim=1)
        x_out = self.propagate(combined_edge_index, x=x)
        x_update = self.alpha * x + (1 - self.alpha) * x_out 
        return x_update
    def message(self, x_j):
        return x_j
    
    ################# Distance-weighted region aggregation ################
    def aggregate(self, inputs, index, edge_index,x):
        out = torch.zeros_like(x, device=inputs.device)
        src_nodes = edge_index[0]  
        tgt_nodes = edge_index[1]  
        for i in range(index.size(0)):
            distance = torch.tensor(src_nodes[i] - tgt_nodes[i])  # 计算源节点和目标节点的距离
            weight = 1.0 / (distance + 1e-6)  
            out[index[i]] += weight * inputs[i]  # 加权累加到目标节点的特征
        return out
                  
    # All connection relation
    def get_edge_index_from_indices(self, indices_list, mode='stem'):
        start_cat_total, end_cat_total = [], []
        for indices in indices_list:
            indices = sorted(indices)
            for i in range(len(indices) - 1):
                u, v = indices[i], indices[i + 1]
                start_cat_total.extend([u, v])
                end_cat_total.extend([v, u])
            if mode == 'stem':
                half_len = len(indices) // 2
                reversed_front = indices[:half_len][::-1]
                back = indices[half_len:]
                start_cat_total.extend(reversed_front + back)
                end_cat_total.extend(back + reversed_front)
        edge_index_list = [start_cat_total, end_cat_total]
        edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long)    
        return edge_index_tensor
    
    # Region sequential connection relationship
    def get_edge_index_from_indices_region(self, indices_list):
        start_cat_total, end_cat_total = [], []
        for indices in indices_list:
            indices = sorted(indices)
            for i in range(len(indices) - 1):
                u, v = indices[i], indices[i + 1]
                start_cat_total.extend([u, v])
                end_cat_total.extend([v, u])
        edge_index_list = [start_cat_total, end_cat_total]
        edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long)    
        return edge_index_tensor

    # All interconnect relation
    def get_edge_index_from_indices_interconnect(self, indices_list):
        start_cat_total, end_cat_total = [], []
        for indices in indices_list:
            indices = sorted(indices)
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    u, v = indices[i], indices[j]
                    start_cat_total.extend([u, v])
                    end_cat_total.extend([v, u])
        edge_index_list = [start_cat_total, end_cat_total]
        edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long)
        return edge_index_tensor

class SublistAndIndexCalculator:
    def __init__(self):
        self.stem_indices = set()
        self.loop_indices = set()
        self.stem_only_indices = set()
        self.stem_indices_per_triple = []
        self.loop_indices_per_triple = []
        self.stem_only_indices_per_triple = []
        self.Sublist = []
        self.central = []  

    def process_and_calculate(self, edge_index):
        row1, row2 = edge_index
        max_edge_index_value = edge_index.max().item()
        avg_row = (row1 + row2) / 2
        symmetry = torch.stack([avg_row, row1, row2], dim=0)
        len_stem = defaultdict(int)
        unique_values, counts = torch.unique(symmetry[0], return_counts=True)
        for value, count in zip(unique_values.tolist(), counts.tolist()):
            len_stem[value] = count
        keys_to_remove = {key for key, count in len_stem.items() if count == 2}
        keep_mask = ~torch.tensor([value in keys_to_remove for value in symmetry[0].tolist()])
        symmetry = symmetry[:, keep_mask]
        valid_mask = symmetry[1] < symmetry[2]
        symmetry = symmetry[:, valid_mask]
        difference = torch.abs(symmetry[1] - symmetry[2])
        valid_mask = difference != 1
        symmetry = symmetry[:, valid_mask]
        len_stem = defaultdict(int)
        for value in symmetry[0].tolist():
            len_stem[value] += 1
        sorted_indices = torch.argsort(symmetry[0])
        sorted_symmetry = symmetry[:, sorted_indices]
        min_dist = {}
        group = defaultdict(list)
        for value, row1, row2 in zip(sorted_symmetry[0].tolist(), sorted_symmetry[1].tolist(), sorted_symmetry[2].tolist()):
            group[value].append(row1)
        for value, rows in group.items():
            min_dist[value] = abs(value - max(rows))
        self.Sublist = []
        for key, length in len_stem.items():
            min_dist_value = min_dist.get(key, float('inf'))
            if (
                # humanbestdist
                # 2 <= length <= 5
                # and (2 <= min_dist_value <= 10 or 25 <= min_dist_value <= 35) 
                # circrnabestdist
                # 2 <= length <= 6
                # and (2 <= min_dist_value <= 9 or 27 <= min_dist_value <= 34) 
                # humancircrnadist
                2 <= length <= 5
                and (2 <= min_dist_value <= 6)  
                and key - length - min_dist_value + 1 > 0
                and key + length + min_dist_value - 1 < max_edge_index_value  
            ):
                self.Sublist.append([key, length, min_dist_value])

        self.calculate_indices()

    def calculate_indices(self):
        self.stem_indices_per_triple = []
        self.loop_indices_per_triple = []
        self.stem_only_indices_per_triple = []
        self.central = []  

        for triple in self.Sublist:
            central, length, min_dist = triple
            stem_start = int(central - length - min_dist + 1)
            stem_end = int(central + length + min_dist - 1)
            stem_range = range(stem_start, stem_end + 1)
            
            loop_start = int(central - min_dist + 1)
            loop_end = int(central + min_dist - 1)
            loop_range = range(loop_start, loop_end + 1)
            stem_range_set = set(stem_range)
            loop_range_set = set(loop_range)
            stem_only_range_set = stem_range_set - loop_range_set

            self.stem_indices_per_triple.append(list(stem_range))
            self.loop_indices_per_triple.append(list(loop_range))
            self.stem_only_indices_per_triple.append(list(stem_only_range_set))

            self.stem_indices.update(stem_range_set)
            self.loop_indices.update(loop_range_set)
            self.stem_only_indices.update(stem_only_range_set)

            self.central.append(central)  

    def get_indices(self):
        # print(self.loop_indices_per_triple,self.stem_only_indices_per_triple, self.central)
        return {
            'loop_indices': self.loop_indices_per_triple,
            'stem_indices': self.stem_only_indices_per_triple,
            'central': self.central,  
        }

    def extract_edge_index(self, E):
        bs, n, _, _ = E.shape
        edge_index_list = [torch.nonzero(E[b].sum(dim=-1), as_tuple=False).t().long() for b in range(bs)]
        return edge_index_list
