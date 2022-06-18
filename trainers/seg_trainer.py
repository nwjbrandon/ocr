import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.losses import dice_loss


class Trainer:
    def __init__(self, cfg, model, scheduler, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.criterion1 = nn.CrossEntropyLoss()

        self.n_classes = cfg["dataset"]["n_classes"]

        self.epochs = cfg["train"]["epochs"]
        self.device = cfg["train"]["device"]
        self.loss_file = cfg["train"]["loss_file"]
        self.model_file = cfg["train"]["model_file"]
        self.ckpt_freq = cfg["train"]["ckpt_freq"]
        self.eval_step = cfg["train"]["eval_step"]

        self.step = 0
        self.loss = {"train": [], "val": []}

    def train(self, train_dataloader, val_dataloader):
        for epoch in range(self.epochs):
            self._epoch_train(train_dataloader)
            self._epoch_eval(val_dataloader)

            log = "Epoch: {}/{}, Train Loss={}, Val Loss={}".format(
                epoch + 1,
                self.epochs,
                np.round(self.loss["train"][-1], 10),
                np.round(self.loss["val"][-1], 10),
            )
            print(log)
            with open(self.loss_file, "a") as f:
                f.write(f"{log}\n")

            # reducing LR if no improvement
            self.scheduler.step(self.loss["val"][-1])

            # saving model
            if (epoch + 1) % self.ckpt_freq == 0:
                torch.save(
                    self.model.state_dict(), f"{self.model_file}_{epoch+1}.pth"
                )

        torch.save(self.model.state_dict(), "pytorch_model_last")
        return self.model

    def _epoch_train(self, train_dataloader):
        self.model.train()
        running_loss = []

        for i, data in enumerate(tqdm(train_dataloader), 0):
            image_inp = data["image_inp"].float()
            mask_gt = data["mask_gt"]

            image_inp = image_inp.to(self.device)
            mask_gt = mask_gt.to(self.device)

            self.optimizer.zero_grad()

            mask_pred = self.model(image_inp)

            losses = self.criterion(mask_pred, mask_gt)

            losses.backward()
            self.optimizer.step()
            running_loss.append(losses.item())

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

    def _epoch_eval(self, epoch, dataloader):
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader), 0):
                image_inp = data["image_inp"].float()
                mask_gt = data["mask_gt"]

                image_inp = image_inp.to(self.device)
                mask_gt = mask_gt.to(self.device)

                mask_pred = self.model(image_inp)

                losses = self.criterion(mask_pred, mask_gt)

                running_loss.append(losses.item())

        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)

    def criterion(self, mask_pred, mask_gt):
        loss = self.criterion1(mask_pred, mask_gt) + dice_loss(
            F.softmax(mask_pred, dim=1).float(),
            F.one_hot(mask_gt, self.n_classes).permute(0, 3, 1, 2).float(),
            multiclass=True,
        )
        return loss
