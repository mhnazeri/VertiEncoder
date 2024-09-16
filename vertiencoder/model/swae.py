"""Courtesy of https://github.com/eifuentes/swae-pytorch/blob/master/swae/models/mnist.py"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Encoder from Original Paper's Keras based Implementation.

    Args:
        init_num_filters (int): initial number of filters from encoder image channels
        lrelu_slope (float): positive number indicating LeakyReLU negative slope
        inter_fc_dim (int): intermediate fully connected dimensionality prior to embedding layer
        embedding_dim (int): embedding dimensionality
    """

    def __init__(self, in_channels: int = 1, hidden_dims=None, latent_dim: int = 4):
        super().__init__()

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_z = nn.Linear(2048, latent_dim)

    def forward(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        z = self.fc_z(result)
        return z


class Decoder(nn.Module):
    """MNIST Decoder from Original Paper's Keras based Implementation.

    Args:
        init_num_filters (int): initial number of filters from encoder image channels
        lrelu_slope (float): positive number indicating LeakyReLU negative slope
        inter_fc_dim (int): intermediate fully connected dimensionality prior to embedding layer
        embedding_dim (int): embedding dimensionality
    """

    def __init__(self, out_channels: int = 1, hidden_dims=None, latent_dim: int = 4):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 2 * 2)
        hidden_dims.reverse()
        modules = []

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[0],
                    hidden_dims[0 + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(hidden_dims[0 + 1]),
                nn.LeakyReLU(),
            )
        )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[1],
                    hidden_dims[1 + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(hidden_dims[1 + 1]),
                nn.LeakyReLU(),
            )
        )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[2],
                    hidden_dims[2 + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(hidden_dims[2 + 1]),
                nn.LeakyReLU(),
            )
        )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[3],
                    hidden_dims[3 + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(hidden_dims[3 + 1]),
                nn.LeakyReLU(),
            )
        )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(
                hidden_dims[-1], out_channels=out_channels, kernel_size=3, padding=1
            ),
            nn.LeakyReLU(negative_slope=0.3),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                out_channels, out_channels=out_channels, kernel_size=3, padding=1
            ),
        )

    def forward(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result


class SWAutoencoder(nn.Module):
    """Sliced Wasserstein Autoencoder from Original Paper's Keras based Implementation.

    Args:
        init_num_filters (int): initial number of filters from encoder image channels
        lrelu_slope (float): positive number indicating LeakyReLU negative slope
        inter_fc_dim (int): intermediate fully connected dimensionality prior to embedding layer
        embedding_dim (int): embedding dimensionality
    """

    def __init__(self, in_channels: int = 1, hidden_dims=None, latent_dim: int = 4):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dims, latent_dim)
        self.decoder = Decoder(in_channels, hidden_dims, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class SWNTAutoencoder(nn.Module):
    """Sliced Wasserstein Autoencoder from Original Paper's Keras based Implementation for next token prediction.

    Args:
        init_num_filters (int): initial number of filters from encoder image channels
        lrelu_slope (float): positive number indicating LeakyReLU negative slope
        inter_fc_dim (int): intermediate fully connected dimensionality prior to embedding layer
        embedding_dim (int): embedding dimensionality
    """

    def __init__(self, in_channels: int = 1, hidden_dims=None, latent_dim: int = 4):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dims, latent_dim)
        self.decoder = Decoder(in_channels, hidden_dims, latent_dim)
        self.action_encoder = nn.Sequential(
            nn.Linear(2, 2 * latent_dim),
            nn.LayerNorm(2 * latent_dim),
            nn.ReLU(),
            nn.Linear(2 * latent_dim, latent_dim),
            nn.ReLU(),
        )
        self.fc_z = nn.Linear(2 * latent_dim, latent_dim)

    def forward(self, x, action):
        z = self.encoder(x)
        action_z = self.action_encoder(action)
        z = self.fc_z(torch.cat([z, action_z], dim=1))
        return self.decoder(z), z

    def generate(
        self,
        patches: torch.Tensor,
        actions: torch.Tensor,
        future_actions: torch.Tensor,
        timesteps: int = 15,
    ) -> torch.Tensor:
        next_patch, _ = self(patches, actions)
        for i in range(timesteps - 1):
            patches = next_patch
            actions = future_actions[:, i]
            next_patch, _ = self(patches, actions)

        return next_patch
