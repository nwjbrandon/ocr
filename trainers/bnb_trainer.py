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
        for _ in train_dataloader:
            pass
        for _ in val_dataloader:
            pass
