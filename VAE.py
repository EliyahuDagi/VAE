"""NICE model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, latent_dim, device):
        """Initialize a VAE.

        Args:
            latent_dim: dimension of embedding
            device: run on cpu or gpu
        """
        super(Model, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 1, 2),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7
        )

        self.mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.upsample = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  64,  14,  14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 1, 2),  # B, 1, 28, 28
            nn.Sigmoid()
        )
        self.bce = nn.BCELoss(reduction='none')

    def sample(self, sample_size, mu=None, logvar=None):
        """
        :param sample_size: Number of samples
        :param mu: z mean, None for prior (init with zeros)
        :param logvar: z logstd, None for prior (init with zeros)
        :return:
        """
        if mu is None:
            mu = torch.zeros((sample_size, self.latent_dim)).to(self.device)
        if logvar is None:
            logvar = torch.zeros((sample_size, self.latent_dim)).to(self.device)
        x = self.upsample(self.z_sample(mu=mu, log_var=logvar)).reshape(-1, 64, 7, 7)
        return self.decoder(x)

    @staticmethod
    def z_sample(mu, log_var):
        eps = torch.randn_like(mu)
        return mu + torch.exp(log_var * 0.5) * eps

    def loss(self, x, recon, mu, logvar):
        reconstruct_term = torch.mean(self.bce(recon, x), dim=[1, 2, 3])
        kl_term = torch.mean(-0.5 * (1 + logvar - (mu * mu) - torch.exp(logvar)), dim=1)
        return torch.mean(reconstruct_term + kl_term)

    def forward(self, x):
        encoded = nn.Flatten()(self.encoder(x))
        mu = self.mu(encoded)
        log_var = self.logvar(encoded)
        z = self.z_sample(mu=mu, log_var=log_var)
        z = self.upsample(z).reshape(-1, 64, 7, 7)
        reconstruction = self.decoder(z)
        return mu, log_var, reconstruction


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.n_count = 0
        self.sum = 0

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res