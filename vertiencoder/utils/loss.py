"""Courtesy of https://github.com/eifuentes/swae-pytorch/blob/master/swae/trainer.py"""

import torch
import torch.nn.functional as F
from torch import distributions as dist


def sliced_wasserstein_distance(
    recons,
    input,
    z,
    reg_weight,
    wasserstein_de,
    num_projections,
    projection_dist,
    latent_dim,
) -> dict:

    batch_size = input.size(0)
    bias_corr = batch_size * (batch_size - 1)
    reg_weight = reg_weight / bias_corr

    recons_loss_l2 = F.mse_loss(recons, input)
    # recons_loss_l1 = F.l1_loss(recons, input)

    swd_loss = compute_swd(
        z, wasserstein_de, reg_weight, latent_dim, num_projections, projection_dist
    )

    loss = recons_loss_l2 + swd_loss
    return {
        "loss": loss,
        "Reconstruction_Loss": recons_loss_l2,  # + recons_loss_l1),
        "SWD": swd_loss,
    }


def get_random_projections(
    latent_dim: int, num_samples: int, proj_dist
) -> torch.Tensor:
    """
    Returns random samples from latent distribution's (Gaussian)
    unit sphere for projecting the encoded samples and the
    distribution samples.

    :param latent_dim: (Int) Dimensionality of the latent space (D)
    :param num_samples: (Int) Number of samples required (S)
    :return: Random projections from the latent unit sphere
    """
    if proj_dist == "normal":
        rand_samples = torch.randn(num_samples, latent_dim)
    elif proj_dist == "cauchy":
        rand_samples = (
            dist.Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
            .sample((num_samples, latent_dim))
            .squeeze()
        )
    else:
        raise ValueError("Unknown projection distribution.")

    rand_proj = rand_samples / rand_samples.norm(dim=1).view(-1, 1)
    return rand_proj  # [S x D]


def compute_swd(
    z: torch.Tensor, p: float, reg_weight: float, latent_dim, num_projections, proj_dist
) -> torch.Tensor:
    """
    Computes the Sliced Wasserstein Distance (SWD) - which consists of
    randomly projecting the encoded and prior vectors and computing
    their Wasserstein distance along those projections.

    :param z: Latent samples # [N  x D]
    :param p: Value for the p^th Wasserstein distance
    :param reg_weight:
    :return:
    """
    prior_z = torch.randn_like(z)  # [N x D]
    device = z.device

    proj_matrix = (
        get_random_projections(
            latent_dim, num_samples=num_projections, proj_dist=proj_dist
        )
        .transpose(0, 1)
        .to(device)
    )

    latent_projections = z.matmul(proj_matrix)  # [N x S]
    prior_projections = prior_z.matmul(proj_matrix)  # [N x S]

    # The Wasserstein distance is computed by sorting the two projections
    # across the batches and computing their element-wise l2 distance
    w_dist = (
        torch.sort(latent_projections.t(), dim=1)[0]
        - torch.sort(prior_projections.t(), dim=1)[0]
    )
    w_dist = w_dist.pow(p)
    return reg_weight * w_dist.mean()


################## old loss
# def rand_cirlce2d(batch_size):
#     """ This function generates 2D samples from a filled-circle distribution in a 2-dimensional space.
#
#         Args:
#             batch_size (int): number of batch samples
#
#         Return:
#             torch.Tensor: tensor of size (batch_size, 2)
#     """
#     r = np.random.uniform(size=(batch_size))
#     theta = 2 * np.pi * np.random.uniform(size=(batch_size))
#     x = r * np.cos(theta)
#     y = r * np.sin(theta)
#     z = np.array([x, y]).T
#     return torch.from_numpy(z).type(torch.FloatTensor)
#
#
# def rand_projections(embedding_dim, num_samples=50):
#     """This function generates `num_samples` random samples from the latent space's unit sphere.
#
#         Args:
#             embedding_dim (int): embedding dimensionality
#             num_samples (int): number of random projection samples
#
#         Return:
#             torch.Tensor: tensor of size (num_samples, embedding_dim)
#     """
#     projections = [w / np.sqrt((w**2).sum())  # L2 normalization
#                    for w in np.random.normal(size=(num_samples, embedding_dim))]
#     projections = np.asarray(projections)
#     return torch.from_numpy(projections).type(torch.FloatTensor)
#
#
# def _sliced_wasserstein_distance(encoded_samples,
#                                  distribution_samples,
#                                  num_projections=50,
#                                  p=2,
#                                  device='cpu'):
#     """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.
#
#         Args:
#             encoded_samples (toch.Tensor): tensor of encoded training samples
#             distribution_samples (torch.Tensor): tensor of drawn distribution training samples
#             num_projections (int): number of projections to approximate sliced wasserstein distance
#             p (int): power of distance metric
#             device (torch.device): torch device (default 'cpu')
#
#         Return:
#             torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
#     """
#     # derive latent space dimension size from random samples drawn from latent prior distribution
#     embedding_dim = distribution_samples.size(1)
#     # generate random projections in latent space
#     projections = rand_projections(embedding_dim, num_projections).to(device)
#     # calculate projections through the encoded samples
#     encoded_projections = encoded_samples.matmul(projections.transpose(0, 1))
#     # calculate projections through the prior distribution random samples
#     distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1)))
#     # calculate the sliced wasserstein distance by
#     # sorting the samples per random projection and
#     # calculating the difference between the
#     # encoded samples and drawn random samples
#     # per random projection
#     wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
#                             torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
#     # distance between latent space prior and encoded distributions
#     # power of 2 by default for Wasserstein-2
#     wasserstein_distance = torch.pow(wasserstein_distance, p)
#     # approximate mean wasserstein_distance for each projection
#     return wasserstein_distance.mean()
#
#
# def sliced_wasserstein_distance(encoded_samples,
#                                 distribution_fn=rand_cirlce2d,
#                                 num_projections=50,
#                                 p=2,
#                                 device='cpu'):
#     """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.
#
#         Args:
#             encoded_samples (toch.Tensor): tensor of encoded training samples
#             distribution_samples (torch.Tensor): tensor of drawn distribution training samples
#             num_projections (int): number of projections to approximate sliced wasserstein distance
#             p (int): power of distance metric
#             device (torch.device): torch device (default 'cpu')
#
#         Return:
#             torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
#     """
#     # derive batch size from encoded samples
#     batch_size = encoded_samples.size(0)
#     # draw random samples from latent space prior distribution
#     z = distribution_fn(batch_size).to(device)
#     # approximate mean wasserstein_distance between encoded and prior distributions
#     # for each random projection
#     swd = _sliced_wasserstein_distance(encoded_samples, z,
#                                        num_projections, p, device)
#     return swd
