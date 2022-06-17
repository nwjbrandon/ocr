import argparse

import torch
from torch.utils.data import DataLoader

from utils.data import collate_fn
from utils.io import import_module, load_config


def main():
    parser = argparse.ArgumentParser(description="Character OCR")
    parser.add_argument(
        "--cfg",
        type=str,
        default="cfgs/text_detection.yml",
        help="specify config file in cfgs",
    )
    parser.add_argument(
        "--mode", type=str, default="train", help="specify mode to run"
    )

    args = parser.parse_args()
    print(args)
    cfg = args.cfg
    # mode = args.mode

    cfg = load_config(cfg)

    # Create model
    Model = import_module(cfg["model"]["model"])
    model = Model(cfg)
    model = model.to(cfg["train"]["device"])
    if cfg["model"]["ckpt"] is not None:
        print("Loading:", cfg["model"]["ckpt"])
        model.load_state_dict(
            torch.load(
                cfg["model"]["ckpt"],
                map_location=torch.device(cfg["train"]["device"]),
            )
        )

    # Create optimizer and schedular
    Optimizer = import_module(cfg["train"]["optimizer"])
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Optimizer(params, lr=cfg["train"]["lr"])
    Scheuler = import_module(cfg["train"]["scheduler"])
    scheduler = Scheuler(
        optimizer=optimizer,
        factor=cfg["train"]["scheduler_factor"],
        patience=cfg["train"]["scheduler_patience"],
        verbose=cfg["train"]["scheduler_verbose"],
        threshold=cfg["train"]["scheduler_threshold"],
    )

    # Create dataset
    Dataset = import_module(cfg["dataset"]["dataset"])
    train_dataset = Dataset(cfg, is_train=True)
    val_dataset = Dataset(cfg, is_train=False)
    train_dataloader = DataLoader(
        train_dataset,
        cfg["train"]["batch_size"],
        shuffle=cfg["train"]["shuffle"],
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn if cfg["dataset"]["use_collate"] else None,
    )
    val_dataloader = DataLoader(
        val_dataset,
        cfg["train"]["batch_size"],
        shuffle=cfg["train"]["shuffle"],
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn if cfg["dataset"]["use_collate"] else None,
    )

    # Train model
    Trainer = import_module(cfg["train"]["trainer"])
    trainer = Trainer(cfg, model, scheduler, optimizer)
    trainer.train(train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()