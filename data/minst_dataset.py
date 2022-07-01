import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

matplotlib.use("TKAgg")


class MinstRecDataset(Dataset):
    def __init__(self, cfg, is_train):
        self.is_visualise = cfg["dataset"]["is_visualise"]
        self.img_training_size = cfg["dataset"]["img_training_size"]
        self.n_char = cfg["dataset"]["n_char"]
        self.is_train = is_train

        if self.is_train:
            self.mnist_trainset = datasets.MNIST(
                root="./data",
                train=self.is_train,
                download=True,
                transform=None,
            )
            print("Size of train set", len(self.mnist_trainset))
            self.n_data = cfg["dataset"]["n_data"]

        else:
            self.mnist_trainset = datasets.MNIST(
                root="./data",
                train=self.is_train,
                download=True,
                transform=None,
            )
            print("Size of val set", len(self.mnist_trainset))

        self.n_data = len(self.mnist_trainset)
        self.image_transform = transforms.Compose([transforms.ToTensor(),])

    def __len__(self):
        return self.n_data

    def _create_label(self, text):
        label = [0 for _ in range(self.n_char)]
        for i, char in enumerate(text):
            label[i] = int(char) + 1
        return np.array(label)

    def __getitem__(self, idx):
        paper_size = (30, 300)
        paper = np.zeros(paper_size, dtype=np.uint8)

        # Create sequence of text
        n_char = random.randint(1, self.n_char)
        idx = random.randint(0, len(self.mnist_trainset) - 1)
        img, label = self.mnist_trainset[idx]
        text = str(label)
        imgs = [
            img,
        ]
        for _ in range(1, n_char):
            idx = random.randint(0, len(self.mnist_trainset) - 1)
            img, label = self.mnist_trainset[idx]
            imgs.append(img)
            text += str(label)
        img = np.hstack(imgs)

        # Random translation
        h, w = img.shape[:2]
        H, W = paper.shape[:2]
        padding = 0
        offset_h = random.randint(padding, H - h - padding)
        offset_w = random.randint(padding, W - w - padding)
        paper[offset_h : offset_h + h, offset_w : offset_w + w] = img

        text_gt = self._create_label(text)
        if self.is_visualise:
            _, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(paper)
            ax.set_title(text)
            plt.show()

        paper = Image.fromarray(paper)
        inp = self.image_transform(paper)

        return {"inp": inp, "text_gt": text_gt, "text_length": len(text)}
