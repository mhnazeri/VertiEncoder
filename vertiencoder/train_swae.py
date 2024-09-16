import gc
from pathlib import Path
from datetime import datetime
import sys
import argparse
from functools import partial

try:
    sys.path.append(str(Path(".").resolve()))
except:
    raise RuntimeError("Can't append root directory of the project to the path")

from comet_ml.integration.pytorch import log_model, watch
from rich import print
import numpy as np
from tqdm import tqdm
from icecream import ic, install
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torcheval.metrics import PeakSignalNoiseRatio as PSNR

from model.swae import SWAutoencoder
from model.dataloader import TvertiDatasetAE
from utils.nn import check_grad_norm, init_weights, op_counter, init_optimizer
from utils.io import save_checkpoint, load_checkpoint
from utils.helpers import get_conf, timeit, init_logger, init_device, fix_seed
from utils.loss import sliced_wasserstein_distance


class Learner:
    def __init__(self, cfg_dir: str):
        # load config file
        self.cfg = get_conf(cfg_dir)
        # set the name for the model
        self.cfg.directory.model_name = (
            f"{self.cfg.logger.experiment_name}-{self.cfg.model.latent_dim}D"
        )
        self.cfg.directory.model_name += f"-{datetime.now():%m-%d-%H-%M}"
        self.cfg.logger.experiment_name = self.cfg.directory.model_name
        self.cfg.directory.save = str(
            Path(self.cfg.directory.save) / self.cfg.directory.model_name
        )
        # if debugging True, set a few rules
        if self.cfg.train_params.debug:
            install()
            ic.enable()
            ic.configureOutput(prefix=lambda: f"{datetime.now():%H:%M:%S} |> ")
            torch.autograd.set_detect_anomaly(True)
            self.cfg.logger.disabled = True
        else:
            ic.disable()
            torch.autograd.set_detect_anomaly(True)
        # initialize the logger and the device
        self.logger = init_logger(self.cfg)
        self.device = init_device(self.cfg)
        # fix the seed for reproducibility
        fix_seed(self.cfg.train_params.seed)
        # creating dataset interface and dataloader for trained data
        self.data, self.val_data = self.init_dataloader()
        # create model and initialize its weights and move them to the device
        self.model = self.init_model()
        if self.cfg.train_params.compile:
            self.model = torch.compile(self.model)
        # log the model gradients, weights, and activations in comet
        watch(self.model)
        self.logger.log_code(folder="./vertiencoder/model/")
        # initialize the optimizer
        self.optimizer, self.scheduler = init_optimizer(
            self.cfg, self.model.parameters(), self.cfg.train_params.optimizer
        )
        num_params = [x.numel() for x in self.model.parameters()]
        trainable_params = [
            x.numel() for x in self.model.parameters() if x.requires_grad
        ]
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} - Number of parameters: {sum(num_params) / 1e6:.2f}M"
        )
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} - Number of trainable parameters: {sum(trainable_params) / 1e6:.2f}M"
        )
        # define loss function
        self.criterion = partial(
            sliced_wasserstein_distance,
            reg_weight=self.cfg.swloss.reg_weight,
            wasserstein_de=self.cfg.swloss.wasserstein_deg,
            num_projections=self.cfg.swloss.num_projections,
            projection_dist=self.cfg.swloss.projection_dist,
            latent_dim=self.cfg.model.latent_dim,
        )
        self.psnr = PSNR()
        # if resuming, load the checkpoint
        self.if_resume()

    def train(self):
        """Trains the model"""
        while self.epoch <= self.cfg.train_params.epochs:
            running_loss = []

            bar = tqdm(
                self.data,
                desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, training: ",
            )
            for data in bar:
                self.iteration += 1
                (loss_data), t_train = self.forward_batch(data)
                t_train /= self.data.batch_size
                running_loss.append(loss_data["loss"])

                bar.set_postfix(
                    loss=loss_data["loss"],
                    Grad_Norm=loss_data["grad_norm"],
                    Reconstruction_Loss=loss_data["Reconstruction_Loss"],
                    SWD=loss_data["SWD"],
                    Time=t_train,
                )

                self.logger.log_metrics(
                    {
                        "batch_loss": loss_data["loss"],
                        "grad_norm": loss_data["grad_norm"],
                        "Reconstruction_Loss": loss_data["Reconstruction_Loss"],
                        "SWD": loss_data["SWD"],
                    },
                    epoch=self.epoch,
                    step=self.iteration,
                )
                self.logger.log_image(
                    loss_data["samples"],
                    f"train_E{self.epoch}",
                    step=self.iteration,
                    image_channels="first",
                )

            bar.close()
            self.scheduler.step()

            # validate on val set
            val_loss, t = self.validate()
            t /= len(self.val_data.dataset)

            # average loss for an epoch
            self.e_loss.append(np.mean(running_loss))  # epoch loss
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} Epoch {self.epoch:03}, "
                + f"Iteration {self.iteration:05} summary: train Loss: "
                + f"[green]{self.e_loss[-1]:.2f}[/green] \t| Val loss: [red]{val_loss:.2f}[/red] "
                + f"\t| PSNR: [red]{self.psnr.compute().item():.2f}[/red] "
                f"\t| time: {t:.3f} seconds\n"
            )

            self.logger.log_metrics(
                {
                    "train_loss": self.e_loss[-1],
                    "val_loss": val_loss,
                    "time": t,
                    "PSNR": self.psnr.compute().item(),
                },
                epoch=self.epoch,
                step=self.iteration,
            )

            if self.epoch % self.cfg.train_params.save_every == 0 or (
                self.psnr.compute().item() > self.best
                and self.epoch >= self.cfg.train_params.start_saving_best
            ):
                self.save()

            gc.collect()
            self.epoch += 1

        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Training is DONE!")

    @timeit
    def forward_batch(self, patch):
        """Forward pass of a batch"""
        self.model.train()
        patch = patch.to(self.device)
        # forward, backward
        recon_patch, z = self.model(patch)
        loss = self.criterion(recon_patch, patch, z)
        self.optimizer.zero_grad()
        loss["loss"].backward()
        # gradient clipping
        if self.cfg.train_params.grad_clipping > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.train_params.grad_clipping
            )
        # update
        self.optimizer.step()
        # check grad norm for debugging
        grad_norm = check_grad_norm(self.model)
        log_samples = make_grid(
            [
                make_grid(
                    [gt, pred],
                    nrow=2,
                    value_range=(-1, 1),
                    normalize=True,
                    scale_each=True,
                )
                for gt, pred in zip(patch[:64].cpu(), recon_patch[:64].cpu().detach())
            ],
            nrow=8,
        )

        return {
            "loss": loss["loss"].detach().item(),
            "grad_norm": grad_norm,
            "SWD": loss["SWD"].item(),
            "encode": z.detach().detach(),
            "Reconstruction_Loss": loss["Reconstruction_Loss"].item(),
            "samples": log_samples,
        }

    @timeit
    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = []
        self.psnr.reset()
        bar = tqdm(
            self.val_data,
            desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, validating",
        )
        for patch in bar:
            # move data to device
            patch = patch.to(self.device)
            # forward, backward
            recon_patch, z = self.model(patch)
            loss = self.criterion(recon_patch, patch, z)
            self.psnr.update(recon_patch, patch)
            log_samples = make_grid(
                [
                    make_grid([gt, pred], nrow=2, value_range=(-1, 1), normalize=True)
                    for gt, pred in zip(patch[:64].cpu(), recon_patch[:64].cpu())
                ],
                nrow=8,
            )
            running_loss.append(loss["loss"].item())
            bar.set_postfix(loss=loss["loss"].item(), PSNR=self.psnr.compute().item())
            self.logger.log_image(
                log_samples,
                f"val_E{self.epoch}",
                step=self.iteration,
                image_channels="first",
            )
        bar.close()
        # average loss
        loss = np.mean(running_loss)

        return loss

    def init_model(self):
        """Initializes the model"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the model!")
        model = SWAutoencoder(**self.cfg.model)

        if (
            "cuda" in str(self.device)
            and self.cfg.train_params.device.split(":")[1] == "a"
        ):
            model = torch.nn.DataParallel(model)

        model.apply(init_weights(**self.cfg.init_model))
        model = model.to(device=self.device)
        return model

    def init_dataloader(self):
        """Initializes the dataloaders"""
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the train and val dataloaders!"
        )
        # defining the dataset interface
        dataset = TvertiDatasetAE(**self.cfg.dataset)
        self.cfg.dataset.update(self.cfg.val_dataset)
        val_dataset = TvertiDatasetAE(**self.cfg.dataset)
        # creating dataloader
        data = DataLoader(dataset, **self.cfg.dataloader)

        self.cfg.dataloader.update({"shuffle": False})  # for val dataloader
        val_data = DataLoader(val_dataset, **self.cfg.dataloader)

        # log dataset status
        self.logger.log_parameters(
            {"train_len": len(dataset), "val_len": len(val_dataset)}
        )
        print(
            f"Training consists of {len(dataset)} samples, and validation consists of {len(val_dataset)} samples."
        )

        return data, val_data

    def if_resume(self):
        if self.cfg.logger.resume:
            # load checkpoint
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - LOADING checkpoint!!!")
            save_dir = self.cfg.directory.load
            checkpoint = load_checkpoint(save_dir, self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.epoch = checkpoint["epoch"] + 1
            self.e_loss = checkpoint["e_loss"]
            self.iteration = checkpoint["iteration"] + 1
            self.best = checkpoint["best"]
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} "
                + f"LOADING checkpoint was successful, start from epoch {self.epoch}"
                + f" and loss {self.best}"
            )
        else:
            self.epoch = 1
            self.iteration = 0
            self.best = -np.inf
            self.e_loss = []

        self.logger.set_epoch(self.epoch)

    def save(self, name=None):
        model = self.model
        if isinstance(self.model, torch.nn.DataParallel):
            model = model.module

        checkpoint = {
            "time": str(datetime.now()),
            "epoch": self.epoch,
            "iteration": self.iteration,
            "model": model.state_dict(),
            "encoder": model.encoder.state_dict(),
            "decoder": model.decoder.state_dict(),
            "model_name": type(model).__name__,
            "optimizer": self.optimizer.state_dict(),
            "optimizer_name": type(self.optimizer).__name__,
            "lr_scheduler": self.scheduler.state_dict(),
            "best": self.best,
            "e_loss": self.e_loss,
        }

        if name is None:
            save_name = f"{self.cfg.directory.model_name}-E{self.epoch}"
        else:
            save_name = name

        if self.psnr.compute().item() > self.best:
            self.best = self.psnr.compute().item()
            checkpoint["best"] = self.best
            save_checkpoint(checkpoint, True, self.cfg.directory.save, save_name)
            if self.cfg.logger.upload_model:
                # upload only the current checkpoint
                log_model(self.logger, checkpoint, model_name=save_name)
        else:
            save_checkpoint(checkpoint, False, self.cfg.directory.save, save_name)
            if self.cfg.logger.upload_model:
                # upload only the current checkpoint
                log_model(self.logger, checkpoint, model_name=save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="conf/swae", type=str)
    args = parser.parse_args()
    cfg_path = args.conf
    learner = Learner(cfg_path)
    learner.train()
