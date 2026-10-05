import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models.transformer_model import GraphTransformer
from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition
from diffusion import diffusion_utils
import utils


class DiscreteDenoisingDiffusion(pl.LightningModule):
    """Discrete graph diffusion model (inference only)."""

    def __init__(self, cfg, dataset_infos, graph_features):
        super().__init__()
        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        self.cfg = cfg
        self.T = cfg.model.diffusion_steps

        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = dataset_infos.nodes_dist
        self.dataset_info = dataset_infos
        self.target = 11.5

        self.graph_features = graph_features

        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)

        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                              y_classes=self.ydim_output)
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)

        elif cfg.model.transition == 'marginal':
            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)

            print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.ydim_output) / self.ydim_output)

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        t = noisy_data['t']
        return self.model(X, E, y, node_mask, t)

    @torch.no_grad()
    def sample_batch(self, batch_size: int, num_nodes=None):
        """
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :return: list of [nucleotide_types (n,), edge_types (n, n), zscore (1,)] per sample
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()

        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        z_T = diffusion_utils.sample_discrete_feature_noise(
            limit_dist=self.limit_dist,
            node_mask=node_mask,
            y_init_range=getattr(self, "y_init_range", None),
        )
        X, E, y = z_T.X, z_T.E, z_T.y
        assert (E == torch.transpose(E, 1, 2)).all()

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        target = self.target
        lambda_val = torch.full((batch_size, 1), 0.1, dtype=torch.float32, device=self.device)
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            sampled_s, lambda_val = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask, target, lambda_val)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        samples = []
        for i in range(batch_size):
            n = n_nodes[i]
            nucleotide_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            zscore = y[i].cpu()
            samples.append([nucleotide_types, edge_types, zscore])
        return samples

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask, target, lambda_val):
        """Samples from zs ~ p(zs | zt) with Z-score guidance towards `target`."""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        # Z-score guidance
        Y_target = torch.full((bs, 1), target, dtype=torch.float32, device=self.device)
        Yt_minus_1 = pred.y
        Yt = noisy_data['y_t']
        a = torch.norm(Yt_minus_1 - Y_target, p=2, dim=1)  # distance Y_t-1 -> Y_target
        b = torch.norm(Yt - Yt_minus_1, p=2, dim=1)        # distance Y_t -> Y_t-1
        c = torch.norm(Yt - Y_target, p=2, dim=1)          # distance Y_t -> Y_target
        threshold = 1
        min_lambda = 0.0000001
        max_lambda = 0.9
        X_change = torch.norm(noisy_data["X_t"] - prob_X, p=2, dim=(1, 2), keepdim=True)
        X_change = X_change.squeeze(2)
        lambda_val = torch.where(X_change > threshold,
                                 torch.clamp(lambda_val * 0.1, min=min_lambda, max=max_lambda),
                                 torch.clamp(lambda_val * 1.1, min=min_lambda, max=max_lambda))
        x_M = torch.sqrt(b**2 * (1 - lambda_val) + c**2 * lambda_val - a**2 * lambda_val * (1 - lambda_val))
        x_M = x_M[0:1].T
        b = b.unsqueeze(1)
        mask = (x_M - b) > 0.0001
        deta_y = torch.zeros_like(x_M)
        deta_y[mask] = x_M[mask]
        pred_y = deta_y + Yt

        # The first draw is discarded; it is kept so that seeded runs reproduce the original samples.
        diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)
        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)
        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()
        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=pred_y)
        return out_one_hot.mask(node_mask).type_as(pred_y), lambda_val

    def compute_extra_data(self, noisy_data):
        """ Compute graph features at every sampling step and append them to the network input. """
        graph_features = self.graph_features(noisy_data)
        return utils.PlaceHolder(X=graph_features.X, E=graph_features.E, y=graph_features.y)
