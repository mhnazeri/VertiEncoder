from pathlib import Path
import pickle
from collections import defaultdict

import numpy as np
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset

from vertiencoder.utils.helpers import to_tensor, read_patch


class TvertiDatasetBase(Dataset):

    def __init__(
        self,
        root: str,
        stats: str,
        train: bool = True,
        block_size: int = 20,
        f_size: int = 7,
        height_diff: int = 0.5,
        pred_len: int = 1,
    ):
        self.block_size = block_size
        self.height_diff = height_diff
        self.root = Path(root)  # 20 bags
        self.pred_len = pred_len
        with open(self.root, "rb") as f:
            metadata = pickle.load(f)

        with open(stats, "rb") as f:
            self.stats = pickle.load(f)

        self.metadata = defaultdict(list)
        self.f_size = f_size
        for i, bag_name in enumerate(metadata["bag_name"]):
            if len(metadata["data"][i]["cmd_vel"]) < self.f_size:
                continue
            num_samples = len(metadata["data"][i]["cmd_vel"])
            bag_data = defaultdict(list)
            cmd_vel = np.array(metadata["data"][i]["cmd_vel"], dtype=np.float32)
            cmd_vel_filtered = np.zeros(
                (cmd_vel.shape[0] - self.f_size + 1, cmd_vel.shape[1]), dtype=np.float32
            )
            cmd_vel_filtered[:, 0] = np.convolve(
                cmd_vel[:, 0], np.ones(self.f_size) / self.f_size, mode="valid"
            )
            cmd_vel_filtered[:, 1] = np.convolve(
                cmd_vel[:, 1], np.ones(self.f_size) / self.f_size, mode="valid"
            )
            cmd_vel = cmd_vel_filtered

            pose_diff = np.array(metadata["data"][i]["pose_diff"], dtype=np.float32)
            pose_diff_filtered = np.zeros(
                (pose_diff.shape[0] - self.f_size + 1, pose_diff.shape[1]),
                dtype=np.float32,
            )
            for k in range(6):
                pose_diff_filtered[:, k] = np.convolve(
                    pose_diff[:, k], np.ones(self.f_size) / self.f_size, mode="valid"
                )
            pose_diff = pose_diff_filtered / self.stats["pose_diff_max"]
            # trimming data because of convolution
            metadata["data"][i]["footprint"] = metadata["data"][i]["footprint"][
                self.f_size // 2 : -self.f_size // 2
            ]
            metadata["data"][i]["pose"] = metadata["data"][i]["pose"][
                self.f_size // 2 : -self.f_size // 2
            ]
            metadata["data"][i]["dt"] = metadata["data"][i]["dt"][
                self.f_size // 2 : -self.f_size // 2
            ]
            metadata["data"][i]["time"] = metadata["data"][i]["time"][
                self.f_size // 2 : -self.f_size // 2
            ]
            metadata["data"][i]["motor_speed"] = metadata["data"][i]["motor_speed"][
                self.f_size // 2 : -self.f_size // 2
            ]
            num_samples = num_samples - (self.f_size - 1)

            for j in range(num_samples - self.block_size - self.pred_len):
                bag_data["cmd_vel"].append(
                    cmd_vel[j : j + self.block_size + self.pred_len].tolist()
                )
                bag_data["footprint"].append(
                    metadata["data"][i]["footprint"][
                        j : j + self.block_size + self.pred_len
                    ]
                )
                bag_data["pose"].append(
                    metadata["data"][i]["pose"][j : j + self.block_size + self.pred_len]
                )
                bag_data["motor_speed"].append(
                    metadata["data"][i]["motor_speed"][
                        j : j + self.block_size + self.pred_len
                    ]
                )
                bag_data["dt"].append(
                    metadata["data"][i]["dt"][j : j + self.block_size + self.pred_len]
                )
                bag_data["pose_diff"].append(
                    pose_diff[j : j + self.block_size + self.pred_len].tolist()
                )
                bag_data["time"].append(
                    metadata["data"][i]["time"][j : j + self.block_size + self.pred_len]
                )

            self.metadata["cmd_vel"].extend(bag_data["cmd_vel"])
            self.metadata["footprint"].extend(bag_data["footprint"])
            self.metadata["pose"].extend(bag_data["pose"])
            self.metadata["motor_speed"].extend(bag_data["motor_speed"])
            self.metadata["dt"].extend(bag_data["dt"])
            self.metadata["pose_diff"].extend(bag_data["pose_diff"])
            self.metadata["time"].extend(bag_data["time"])

        self.transform = v2.Compose(
            [
                v2.Resize(size=(40, 40), antialias=True),
                v2.ToDtype(torch.float32, scale=False),
            ]
        )

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class TvertiDatasetAE(Dataset):
    def __init__(
        self, root: str, stats: str, train: bool = True, height_diff: float = 0.5
    ):
        self.height_diff = height_diff
        self.root = Path(root)
        with open(root, "rb") as f:
            metadata = pickle.load(f)

        with open(stats, "rb") as f:
            self.stats = pickle.load(f)
            print(f"Loaded stats from {self.stats['name']}!")

        self.metadata = defaultdict(list)
        num_samples = 0
        for i, bag_name in enumerate(metadata["bag_name"]):
            self.metadata["footprint"].extend(metadata["data"][i]["footprint"])
            self.metadata["pose"].extend(metadata["data"][i]["pose"])

        num_samples = len(self.metadata["pose"])

        assert num_samples == len(
            self.metadata["footprint"]
        ), "The number of samples does not match"
        if train:
            self.transform = v2.Compose(
                [
                    v2.Resize(size=(40, 40), antialias=True),
                    v2.ToDtype(torch.float32, scale=False),
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomVerticalFlip(p=0.5),                ]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.Resize(size=(40, 40), antialias=True),
                    v2.ToDtype(torch.float32, scale=False),
                ]
            )

    def __len__(self):
        return len(self.metadata["footprint"])

    def __getitem__(self, idx):
        """Return a sample in the form: (patch, )"""
        patch = self.transform(
            read_patch(
                self.root.parents[0] / self.metadata["footprint"][idx],
                self.metadata["pose"][idx][2],
                self.height_diff,
            )
        )
        return patch


class TvertiDatasetAENextToken(TvertiDatasetBase):
    def __init__(
        self,
        root: str,
        stats: str,
        train: bool = True,
        f_size: int = 7,
        height_diff: int = 0.5,
    ):
        super().__init__(
            root=root, stats=stats, train=train, f_size=f_size, height_diff=height_diff
        )

    def __len__(self):
        return len(self.metadata["cmd_vel"]) - self.block_size - 16

    def __getitem__(self, idx):
        """Return a sample in the form: (patch, next_patch)"""
        patch = self.transform(
            read_patch(
                self.root.parents[0] / self.metadata["footprint"][idx],
                self.metadata["pose"][idx][2],
                self.height_diff,
            )
        )
        patch = (patch - self.stats["footprint_mean"]) / self.stats["footprint_std"]
        # next patch
        next_patch = self.transform(
            read_patch(
                self.root.parents[0]
                / self.metadata["footprint"][idx + self.block_size + 15],
                self.metadata["pose"][idx + self.block_size + 15][2],
                self.height_diff,
            )
        )
        next_patch = (next_patch - self.stats["footprint_mean"]) / self.stats[
            "footprint_std"
        ]
        next_cmd_vel = torch.stack(
            [
                (to_tensor(self.metadata["cmd_vel"][i]) - self.stats["cmd_vel_mean"])
                / self.stats["cmd_vel_std"]
                for i in range(idx + self.block_size, idx + self.block_size + 15)
            ],
            dim=0,
        )
        current_cmd_vel = to_tensor(self.metadata["cmd_vel"][idx])
        return patch, next_patch, current_cmd_vel, next_cmd_vel


class VertiEncoderDataset(TvertiDatasetBase):
    def __init__(
        self,
        root: str,
        stats: str,
        train: bool = True,
        block_size: int = 20,
        f_size: int = 7,
        height_diff: int = 0.5,
    ):
        super().__init__(
            root=root, stats=stats, train=train, f_size=f_size, height_diff=height_diff
        )
        self.block_size = block_size

    def __len__(self):
        return len(self.metadata["pose"])

    def __getitem__(self, idx: int):
        # patch input to the model
        patch = torch.stack(
            [
                (
                    self.transform(
                        read_patch(
                            self.root.parents[0] / self.metadata["footprint"][idx][i],
                            self.metadata["pose"][idx][i][2],
                            self.height_diff,
                        )
                    )
                )
                for i in range(self.block_size)
            ],
            dim=0,
        )
        # next patch the model should predict
        next_patch = self.transform(
            read_patch(
                self.root.parents[0] / self.metadata["footprint"][idx][self.block_size],
                self.metadata["pose"][idx][self.block_size][2],
                self.height_diff,
            )
        )
        cmd_vel = to_tensor(self.metadata["cmd_vel"][idx][: self.block_size])
        pose = to_tensor(self.metadata["pose_diff"][idx][: self.block_size])
        next_pose = to_tensor(self.metadata["pose_diff"][idx][self.block_size])

        return (patch, next_patch, cmd_vel, pose, next_pose)


class TvertiDownStream(TvertiDatasetBase):
    def __init__(
        self,
        root: str,
        stats: str,
        train: bool = True,
        block_size: int = 20,
        task: str = "fkd",
        f_size: int = 7,
        height_diff: int = 0.5,
        pred_len: int = 0,
    ):
        super().__init__(
            root=root,
            stats=stats,
            train=train,
            f_size=f_size,
            height_diff=height_diff,
            pred_len=pred_len,
        )
        self.block_size = block_size
        self.pred_len = pred_len
        self.task = task

    def __len__(self):
        return (
            len(self.metadata["cmd_vel"]) - self.pred_len - self.block_size
        )

    def __getitem__(self, idx: int):
        # patch input to the model
        patch = torch.stack(
            [
                (
                    self.transform(
                        read_patch(
                            self.root.parents[0] / self.metadata["footprint"][idx][i],
                            self.metadata["pose"][idx][i][2],
                            self.height_diff,
                        )
                    )
                )
                for i in range(self.block_size + self.pred_len)
            ],
            dim=0,
        )
        cmd_vel = to_tensor(
            self.metadata["cmd_vel"][idx][: self.block_size + self.pred_len]
        )
        verti_pose = to_tensor(self.metadata["pose_diff"][idx][: self.block_size])
        if self.task == "fkd":
            next_pose = to_tensor(
                self.metadata["pose_diff"][idx][
                    self.block_size : self.block_size + self.pred_len
                ]
            )
            pose = to_tensor(self.metadata["pose_diff"][idx][self.block_size - 1])
            return patch, cmd_vel, verti_pose, (next_pose, pose)
        elif self.task == "bc":
            next_cmd = to_tensor(
                self.metadata["cmd_vel"][idx][
                    self.block_size : self.block_size + self.pred_len
                ]
            )
            return (
                patch,
                cmd_vel,
                verti_pose,
                next_cmd,
            )  # next cmd_vel
        elif self.task == "ikd":
            next_cmd = to_tensor(
                self.metadata["cmd_vel"][idx][
                    self.block_size : self.block_size + self.pred_len
                ]
            )
            next_pose = to_tensor(
                self.metadata["pose_diff"][idx][
                    self.block_size : self.block_size + self.pred_len
                ]
            )
            pose = to_tensor(
                self.metadata["pose_diff"][idx][
                    self.block_size - 1 : self.block_size + self.pred_len - 1
                ]
            )
            return patch, cmd_vel, verti_pose, (next_cmd, next_pose, pose)


class VanillaDownStream(TvertiDatasetBase):
    def __init__(
        self,
        root: str,
        stats: str,
        train: bool = True,
        task: str = "pose",
        f_size: int = 7,
        height_diff: int = 0.5,
    ):
        super().__init__(
            root=root, stats=stats, train=train, f_size=f_size, height_diff=height_diff
        )
        self.task = task

    def __len__(self):
        return len(self.metadata["pose"]) - 1

    def __getitem__(self, idx: int):
        # patch input to the model
        patch = read_patch(
            self.root.parents[0] / self.metadata["footprint"][idx][0],
            self.metadata["pose"][idx][0][2],
            self.height_diff,
        )
        cmd_vel = to_tensor(
            self.metadata["cmd_vel"][idx][0]
        ) 
        if self.task == "fkd":
            pose = to_tensor(self.metadata["pose_diff"][idx][1])
            return patch, cmd_vel, pose
        elif self.task == "bc":
            next_cmd = to_tensor(
                self.metadata["cmd_vel"][idx][1]
            )
            return (
                patch,
                cmd_vel,
                next_cmd,
            )  # next cmd_vel
        elif self.task == "ikd":
            next_cmd = to_tensor(self.metadata["cmd_vel"][idx][1])
            next_pose = to_tensor(self.metadata["pose_diff"][idx][1])
            return patch, cmd_vel, (next_cmd, next_pose)


if __name__ == "__main__":
    pass
