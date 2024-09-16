"""TransformerVertiWheeler to predict next patch underneath the robot"""

from typing import Tuple
import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F
from omegaconf import DictConfig

from vertiencoder.utils.nn import init_weights
from vertiencoder.model.swae import Encoder, Decoder, SWAutoencoder


class Tverti(nn.Module):
    def __init__(self, cfg: DictConfig, patch_encoder: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.transformer.block_size
        self.patch_encoder = patch_encoder
        self.action_encoder = nn.Linear(2, cfg.action_encoder.latent_dim)
        self.pose_encoder = nn.Linear(6, cfg.action_encoder.latent_dim)
        self.action_head = nn.Linear(
            cfg.action_encoder.latent_dim, cfg.action_encoder.latent_dim
        )
        self.pose_head = nn.Linear(
            cfg.action_encoder.latent_dim, cfg.action_encoder.latent_dim
        )
        self.pos = PositionalEncoding(
            cfg.action_encoder.latent_dim, self.block_size * 3 + 1
        )
        transformer_encoder = nn.TransformerEncoderLayer(**cfg.transformer_layer)
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder, cfg.transformer.num_layers
        )
        self.ln = nn.LayerNorm(cfg.action_encoder.latent_dim)
        self.patch_head = nn.Linear(
            cfg.action_encoder.latent_dim, cfg.action_encoder.latent_dim
        )
        self.ctx_token = nn.Parameter(torch.randn(1, 1, cfg.action_encoder.latent_dim))
        self.ctx_head = nn.Linear(
            cfg.action_encoder.latent_dim, cfg.action_encoder.latent_dim
        )
        # a learnable mask
        # self.mask = nn.Parameter(torch.randn(1, 1, cfg.action_encoder.latent_dim))
        # random mask
        self.register_buffer("mask", torch.randn(1, 1, cfg.action_encoder.latent_dim))

    def forward(
        self,
        patches: torch.Tensor,
        actions: torch.Tensor,
        pose: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.action_encoder(actions)
        poses = self.pose_encoder(pose)
        patches = torch.stack(
            [self.patch_encoder(patches[:, i, :]) for i in range(patches.shape[1])],
            dim=1,
        )  # (B, T, L)
        # prepend the ctx token
        b, t, l = poses.size()
        ctx_token = self.ctx_token.expand(b, -1, -1)
        tokens = torch.cat([ctx_token, patches, actions, poses], dim=1)
        if mask is not None:
            masked = self.mask.expand(b, len(mask), -1)
            tokens[torch.arange(b).unsqueeze(1), mask, :] = masked

        tokens = self.pos(tokens)
        pred_tokens = self.transformer_encoder(tokens)
        pred_tokens = self.ln(pred_tokens)
        ctx_pred = self.ctx_head(pred_tokens[:, 0])
        if self.training:
            pred_patch_token = self.patch_head(pred_tokens[:, 1 : self.block_size + 1])
            pred_actions_token = self.action_head(
                pred_tokens[:, self.block_size + 1 : (self.block_size * 2) + 1]
            )
            pred_pose_token = self.pose_head(
                pred_tokens[:, (self.block_size * 2) + 1 :]
            )
            return ctx_pred, pred_patch_token, pred_actions_token, pred_pose_token

        return ctx_pred


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 128, max_len: int = 20):
        super().__init__()
        # the data is in shape (B, T, L)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_encoding


class PatchEmbedding(nn.Module):
    def __init__(
        self, embedding_dim: int = 512, in_channels: int = 1, patch_size: int = 16
    ):
        """Patchify images"""
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        nn.init.kaiming_normal_(self.conv.weight)
        # scale down the effect of projection
        with torch.no_grad():
            self.conv.weight *= 0.1

    def forward(self, x):
        x = self.conv(x)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(b, h * w, c)
        return x.squeeze()


def load_model(cfg):
    """Initializes the model"""
    swae = SWAutoencoder(
        in_channels=cfg.swae.in_channels,
        hidden_dims=cfg.swae.hidden_dims,
        latent_dim=cfg.swae.latent_dim,
    )

    swae_weight = torch.load(cfg.swae_weight, map_location="cpu")["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(swae_weight.items()):
        if k.startswith(unwanted_prefix):
            swae_weight[k[len(unwanted_prefix) :]] = swae_weight.pop(k)
    swae.load_state_dict(swae_weight)
    swae.encoder.eval()
    swae.encoder.requires_grad_(False)

    model = Tverti(cfg, patch_encoder=swae.encoder)

    return model, swae
