import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    def __init__(self, cfg, model, scheduler, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.criterion1 = nn.CTCLoss(
            blank=0, reduction="mean", zero_infinity=True
        )

        self.n_classes = cfg["dataset"]["n_classes"]

        self.epochs = cfg["train"]["epochs"]
        self.device = cfg["train"]["device"]
        self.loss_file = cfg["train"]["loss_file"]
        self.model_file = cfg["train"]["model_file"]
        self.ckpt_freq = cfg["train"]["ckpt_freq"]
        self.eval_step = cfg["train"]["eval_step"]

        self.ckpt_fname = cfg["model"]["ckpt"]

        self.step = 0
        self.loss = {"train": [], "val": []}

    def _get_start_epoch(self):
        if self.ckpt_fname is None:
            start_epoch = 0
        else:
            start_epoch = int(self.ckpt_fname.split(".")[0].split("_")[-1])
        return start_epoch

    def train(self, train_dataloader, val_dataloader):
        start_epoch = self._get_start_epoch()
        print("Start Epoch:", start_epoch)

        for epoch in range(start_epoch, self.epochs):
            self._epoch_train(train_dataloader)
            self._epoch_eval(val_dataloader)
            lr = self.scheduler.get_last_lr()
            log = "Epoch: {}/{}, Train Loss={}, Val Loss={}, LR={}".format(
                epoch + 1,
                self.epochs,
                np.round(self.loss["train"][-1], 10),
                np.round(self.loss["val"][-1], 10),
                np.round(lr[0], 10),
            )
            print(log)
            with open(self.loss_file, "a") as f:
                f.write(f"{log}\n")

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
            image_inp = data["inp"].float()
            text_gt = data["text_gt"]
            text_length_gt = data["text_length"]

            image_inp = image_inp.to(self.device)
            text_gt = text_gt.to(self.device)

            self.optimizer.zero_grad()

            text_pred = self.model(image_inp)

            text_pred = text_pred.permute(1, 0, 2)
            C, B, _ = text_pred.shape
            text_length_pred = torch.IntTensor(B).fill_(C)

            losses = self.criterion(
                text_pred, text_gt, text_length_pred, text_length_gt
            )
            losses.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            self.scheduler.step()

            running_loss.append(losses.item())

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

    def _epoch_eval(self, dataloader):

        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader), 0):
                image_inp = data["inp"].float()
                text_gt = data["text_gt"]
                text_length_gt = data["text_length"]

                image_inp = image_inp.to(self.device)
                text_gt = text_gt.to(self.device)

                text_pred = self.model(image_inp)

                text_pred = text_pred.permute(1, 0, 2)
                C, B, _ = text_pred.shape
                text_length_pred = torch.IntTensor(B).fill_(C)

                losses = self.criterion(
                    text_pred, text_gt, text_length_pred, text_length_gt
                )
                running_loss.append(losses.item())

        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)

    def criterion(self, y_pred, y_train, input_lengths, target_lengths):
        ctc_loss = self.criterion1(
            y_pred, y_train, input_lengths, target_lengths
        )
        return ctc_loss

    def extract_text_pred(self, seq):
        deduplicate_seq = []
        n_tokens = len(seq)
        for i in range(n_tokens):
            token = seq[i]
            if len(deduplicate_seq) == 0:
                deduplicate_seq.append(token)
            else:
                if seq[i - 1] != token:
                    deduplicate_seq.append(token)
        res = []
        for i in range(len(deduplicate_seq)):
            token = deduplicate_seq[i]
            if token != 0:
                res.append(token)
        return np.array(res)

    def extract_text_gt(self, seq):
        res = []
        for i in range(len(seq)):
            token = seq[i]
            if token != 0:
                res.append(token)
        return np.array(res)

    def test(self, dataloader):
        self.model.eval()

        n_corrects = 0

        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader), 0):
                image_inp = data["inp"].float()
                text_gt = data["text_gt"]

                image_inp = image_inp.to(self.device)
                text_gt = text_gt.to(self.device)

                text_pred = self.model(image_inp)

                text_gt = text_gt.cpu().numpy()[0]
                text_pred = text_pred.cpu().numpy()[0]

                text_pred = np.argmax(text_pred, axis=1)

                # print(
                #     "text_gt:", text_gt, "text_pred:", text_pred,
                # )

                text_pred = self.extract_text_pred(text_pred)
                text_gt = self.extract_text_gt(text_gt)

                is_correct = np.array_equal(text_pred, text_gt)
                if is_correct:
                    n_corrects += 1

                # print(
                #     "text_gt:",
                #     text_gt,
                #     "text_pred:",
                #     text_pred,
                #     "correct:",
                #     is_correct,
                # )
                # input("Enter to view next")
        print("Accuracy:", n_corrects / len(dataloader) * 100)
