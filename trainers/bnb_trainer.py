import torch

from trainers.bnb_utils.engine import evaluate, train_one_epoch


class Trainer:
    def __init__(self, cfg, model, scheduler, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

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
            # train for one epoch, printing every 10 iterations
            train_log = train_one_epoch(
                self.model,
                self.optimizer,
                train_dataloader,
                self.device,
                epoch,
                print_freq=100,
            )

            train_loss = train_log.loss.value
            # reducing LR if no improvement
            if self.scheduler is not None:
                self.scheduler.step(train_loss)
            self.loss["val"].append(train_loss)

            evaluate(self.model, val_dataloader, device=self.device)

            # saving model
            if (epoch + 1) % self.ckpt_freq == 0:
                torch.save(
                    self.model.state_dict(), f"{self.model_file}_{epoch+1}.pth"
                )

        torch.save(self.model.state_dict(), "pytorch_model_last")
        return self.model
