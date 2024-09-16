import torch
import torch.nn as nn

from vertiencoder.utils.nn import make_mlp
from vertiencoder.model.tverti import load_model


class FKD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg.fc.dims.append(6)  # append the task output to the end
        self.fc = make_mlp(**cfg.fc)

    def forward(
        self, x: torch.Tensor, cmd: torch.Tensor = None, pose: torch.Tensor = None
    ) -> torch.Tensor:
        return self.fc(x)


class BehaviorCloning(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg.dims.append(2)  # append the task output to the end
        self.fc = make_mlp(**cfg)

    def forward(self, x: torch.Tensor, g_pose: torch.Tensor = None) -> torch.Tensor:
        return self.fc(x)


class IKD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = make_mlp(**cfg.pose)
        cfg.pose.dims[0] = 6
        self.goal_pose = make_mlp(**cfg.pose)
        self.curr_pose = nn.Sequential(
            nn.Linear(6, cfg.pose.dims[-1]),
            nn.LeakyReLU(),
        )
        cfg.fc.dims.insert(0, 3 * cfg.pose.dims[-1])
        cfg.fc.dims.append(2)  # append the task output to the end
        self.fc = make_mlp(**cfg.fc)

    def forward(
        self,
        z: torch.Tensor,
        g_pose: torch.Tensor = None,
        curr_pose: torch.Tensor = None,
    ) -> torch.Tensor:
        z = self.encoder(z)
        g_pose = self.goal_pose(g_pose)
        curr_pose = self.curr_pose(curr_pose)
        pose = torch.cat([z, g_pose, curr_pose], dim=-1)
        return self.fc(pose)
