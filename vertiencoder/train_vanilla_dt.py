import gc
from pathlib import Path
from datetime import datetime
import sys
import argparse
from functools import partial

from omegaconf import OmegaConf

from vertiencoder.utils.helpers import fix_seed

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

from model.vanilla_models import FKD, BehaviorCloning, IKD
from model.dataloader import VanillaDownStream
from utils.nn import check_grad_norm, init_weights, op_counter, init_optimizer
from utils.io import save_checkpoint, load_checkpoint
from utils.helpers import get_conf, timeit, init_logger, init_device, fix_seed


class Learner:
    def __init__(self, cfg_dir: str):
        # load config file
        self.cfg = get_conf(cfg_dir)
        # set the name for the model
        self.cfg.logger.experiment_name += self.cfg.dataset.task
        self.cfg.directory.model_name = (
            f"{self.cfg.logger.experiment_name}-{datetime.now():%m-%d-%H-%M}"
        )
        self.cfg.logger.experiment_name = self.cfg.directory.model_name
        self.cfg.directory.save = str(
            Path(self.cfg.directory.save) / self.cfg.directory.model_name
        )
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} - Starting experiment {self.cfg.logger.experiment_name}!"
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
        # torch.set_default_device(self.device)
        if self.cfg.train_params.seed:
            fix_seed(self.cfg.train_params.seed)
        # creating dataset interface and dataloader for trained data
        self.data, self.val_data = self.init_dataloader()
        # create model and initialize its weights and move them to the device
        self.model = self.init_model(self.cfg.dataset.task)
        self.logger.log_code(folder="./vertiencoder/model/")
        # initialize the optimizer
        self.optimizer, self.scheduler = init_optimizer(
            self.cfg,
            self.model.parameters(),
            self.cfg.train_params.optimizer,
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
        self.criterion = torch.nn.MSELoss()
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
                (loss, grad_norm), t_train = self.forward_batch(data)
                t_train /= self.data.batch_size
                running_loss.append(loss.item())

                bar.set_postfix(loss=loss.item(), Grad_Norm=grad_norm, Time=t_train)

                self.logger.log_metrics(
                    {
                        "batch_loss": loss.item(),
                        "grad_norm": grad_norm,
                    },
                    epoch=self.epoch,
                    step=self.iteration,
                )

            bar.close()
            self.scheduler.step()

            # validate on val set
            val_loss, t = self.validate()
            t /= len(self.val_data.dataset)

            # average loss for an epoch
            self.e_loss.append(np.mean(running_loss))  # epoch loss
            log = (
                (
                    f"{datetime.now():%Y-%m-%d %H:%M:%S} Epoch {self.epoch:03}, "
                    + f"Iteration {self.iteration:05} summary: train Loss: "
                )
                + f"[green]{self.e_loss[-1]:.3f}[/green] \t| Val loss: [red]{val_loss:.3f}[/red] "
            ) + f"\t| time: {t:.3f} seconds\n"
            if self.cfg.dataset.task == "reconstruction":
                log = (
                    log[:-1]
                    + f"\t| PSNR: [red]{self.psnr.compute().item():.3f}[/red]\n"
                )

            print(log)

            metrics = {
                "train_loss": self.e_loss[-1],
                "val_loss": val_loss,
                "time": t,
            }
            self.logger.log_metrics(
                metrics,
                epoch=self.epoch,
                step=self.iteration,
            )

            if self.epoch % self.cfg.train_params.save_every == 0 or (
                self.e_loss[-1] < self.best
                and self.epoch >= self.cfg.train_params.start_saving_best
            ):
                self.save()

            gc.collect()
            self.epoch += 1

        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Training is DONE!")

    @timeit
    def forward_batch(self, batch):
        """Forward pass of a batch"""
        self.model.train()
        patch = batch[0].to(self.device)
        cmd_vel = batch[1].to(self.device)
        if self.cfg.dataset.task == "ikd":
            gt = batch[2][0].to(self.device)
            add_info = batch[2][1].to(self.device)
        else:
            gt = batch[2].to(self.device)
        # forward, backward
        if self.cfg.dataset.task == "ikd":
            pred = self.model(patch, add_info)
        else:
            pred = self.model(patch)
        loss = self.criterion(gt, pred)
        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        if self.cfg.train_params.grad_clipping > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.train_params.grad_clipping
            )
        # update
        self.optimizer.step()
        # check grad norm for debugging
        grad_norm = check_grad_norm(self.model)

        return loss, grad_norm

    @timeit
    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = []
        bar = tqdm(
            self.val_data,
            desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, validating",
        )
        for batch in bar:
            # move data to device
            patch = batch[0].to(self.device)
            cmd_vel = batch[1].to(self.device)
            if self.cfg.dataset.task == "ikd":
                gt = batch[2][0].to(self.device)
                add_info = batch[2][1].to(self.device)
            else:
                gt = batch[2].to(self.device)
            # forward, backward
            if self.cfg.dataset.task == "ikd":
                pred = self.model(patch, add_info)
            else:
                pred = self.model(patch)
            loss = self.criterion(gt, pred)
            running_loss.append(loss.item())
            bar.set_postfix(loss=loss.item())

        bar.close()
        # average loss
        loss = np.mean(running_loss)

        return loss

    def init_model(self, task: str):
        """Initializes the model"""
        print(f"{datetime.now():%H:%M:%S} - INITIALIZING the model!")

        if task == "fkd":
            dt_model = FKD(self.cfg.model.fkd_model)

        elif task == "bc":
            dt_model = BehaviorCloning(self.cfg.model.bc_model)

        elif task == "ikd":
            dt_model = IKD(self.cfg.model.ikd_model)

        else:
            raise Exception(f"{task} is not a valid task!")

        if (
            "cuda" in str(self.device)
            and self.cfg.train_params.device.split(":")[1] == "a"
        ):
            dt_model = torch.nn.DataParallel(dt_model)

        dt_model = dt_model.to(device=self.device)
        return dt_model

    def init_dataloader(self):
        """Initializes the dataloaders"""
        print(
            f"{datetime.now():%H:%M:%S} - Training [green]{self.cfg.dataset.task.upper()}[/green] task!"
        )
        print(
            f"{datetime.now():%H:%M:%S} - INITIALIZING the train and val dataloaders!"
        )
        # defining the dataset interface
        dataset = VanillaDownStream(**self.cfg.dataset)
        self.cfg.dataset.update(self.cfg.val_dataset)
        val_dataset = VanillaDownStream(**self.cfg.dataset)
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
            print(f"{datetime.now():%H:%M:%S} - LOADING checkpoint!!!")
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
                f"{datetime.now():%H:%M:%S} "
                + f"LOADING checkpoint was successful, start from epoch {self.epoch}"
                + f" and loss {self.best}"
            )
        else:
            self.epoch = 1
            self.iteration = 0
            self.best = np.inf
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
            "task": self.cfg.dataset.task,
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

        if self.e_loss[-1] < self.best:
            self.best = self.e_loss[-1]
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
    parser.add_argument("--conf", default="conf/dt", type=str)
    args = parser.parse_args()
    cfg_path = args.conf
    learner = Learner(cfg_path)
    learner.train()
