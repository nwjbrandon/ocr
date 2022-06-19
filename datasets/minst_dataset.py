import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.minst_data import (
    create_affinity_gt,
    create_region_gt,
    generate_data_of_n_characters,
    render_data_on_paper,
    render_multiple_data_on_paper,
    visualize_multiple_centers
)

matplotlib.use("TKAgg")


class MinstDensityDataset(Dataset):
    def __init__(self, cfg, is_train):
        self.is_visualise = cfg["dataset"]["is_visualise"]
        self.img_training_size = cfg["dataset"]["img_training_size"]
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
        self.image_transform = transforms.Compose(
            [transforms.Resize(self.img_training_size), transforms.ToTensor(),]
        )

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):

        (
            paper,
            positions_list,
            bnb_box_list,
            text_list,
        ) = render_multiple_data_on_paper(
            20,
            self.mnist_trainset,
            paper_size=(self.img_training_size, self.img_training_size),
        )
        region_gt = create_region_gt(paper, positions_list)
        affinity_gt = create_affinity_gt(paper, positions_list)

        if self.is_visualise:
            heatmap = visualize_multiple_centers(
                paper, positions_list, bnb_box_list=bnb_box_list
            )
            _, ax = plt.subplots(1, 4, figsize=(20, 20))
            ax[0].imshow(paper)
            ax[0].set_title("Image")
            ax[1].imshow(heatmap)
            ax[1].set_title("Centers")
            ax[2].imshow(region_gt)
            ax[2].set_title("Region")
            ax[3].imshow(affinity_gt)
            ax[3].set_title("Affinity")
            plt.show()

        image = Image.fromarray(paper)
        image_inp = self.image_transform(image)

        return {
            "image_inp": image_inp,
            "region_gt": region_gt,
            "affinity_gt": affinity_gt,
        }


class MinstSegDataset(Dataset):
    def __init__(self, cfg, is_train):
        self.is_visualise = cfg["dataset"]["is_visualise"]
        self.img_training_size = cfg["dataset"]["img_training_size"]
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
        self.image_transform = transforms.Compose(
            [transforms.Resize(self.img_training_size), transforms.ToTensor(),]
        )

    def __len__(self):
        return self.n_data

    def _create_mask(self, image, bnb_box_list):
        mask = np.zeros_like(image, dtype=int)
        mask[image < 100] = 1
        return mask

    def __getitem__(self, idx):

        (
            paper,
            positions_list,
            bnb_box_list,
            text_list,
        ) = render_multiple_data_on_paper(
            20,
            self.mnist_trainset,
            paper_size=(self.img_training_size, self.img_training_size),
        )
        mask = self._create_mask(paper, bnb_box_list)
        if self.is_visualise:
            _, ax = plt.subplots(1, 2, figsize=(20, 20))
            ax[0].imshow(paper)
            ax[0].set_title("Image")
            ax[1].imshow(mask)
            ax[1].set_title("Mask")
            plt.show()

        image = Image.fromarray(paper)
        image_inp = self.image_transform(image)

        return {"image_inp": image_inp, "mask_gt": mask}


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
        self.image_transform = transforms.Compose(
            [transforms.Resize(self.img_training_size), transforms.ToTensor(),]
        )

    def __len__(self):
        return self.n_data

    def _create_label(self, text):
        label = [0 for _ in range(self.n_char)]
        for i, char in enumerate(text):
            label[i] = int(char) + 1
        return np.array(label)

    def __getitem__(self, idx):
        paper_size = (300, 300)
        paper = np.ones(paper_size, dtype=np.uint8) * 255

        n_char = random.randint(0, self.n_char - 1)
        img, text, positions = generate_data_of_n_characters(
            self.mnist_trainset, n_char=n_char
        )
        paper, positions, _, _ = render_data_on_paper(
            img, positions, max_scale=1.2, paper=paper.copy()
        )

        text_gt = self._create_label(text)
        if self.is_visualise:
            _, ax = plt.subplots(1, 1, figsize=(20, 20))
            ax.imshow(paper)
            ax.set_title(text)
            plt.show()

        image = Image.fromarray(paper)
        image_inp = self.image_transform(image)

        return {"image_inp": image_inp, "text_gt": text_gt}


class MinstBnBDataset(Dataset):
    def __init__(self, cfg, is_train, **kwargs):
        super().__init__(**kwargs)
        self.is_visualise = cfg["dataset"]["is_visualise"]
        self.img_training_size = cfg["dataset"]["img_training_size"]
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
        self.image_transform = transforms.Compose(
            [transforms.Resize(self.img_training_size), transforms.ToTensor(),]
        )

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):

        (
            paper,
            positions_list,
            bnb_box_list,
            text_list,
        ) = render_multiple_data_on_paper(
            20,
            self.mnist_trainset,
            paper_size=(self.img_training_size, self.img_training_size),
        )

        if self.is_visualise:
            heatmap = visualize_multiple_centers(
                paper, positions_list, bnb_box_list=bnb_box_list
            )
            _, ax = plt.subplots(1, 2, figsize=(20, 20))
            ax[0].imshow(paper)
            ax[0].set_title("Image")
            ax[1].imshow(heatmap)
            ax[1].set_title("Centers")
            plt.show()

        labels = list()
        areas = list()
        iscrowd = list()
        boxes = list()
        for box in bnb_box_list:
            tl, br = box
            x1, y1 = float(tl[0]), float(tl[1])
            x2, y2 = float(br[0]), float(br[1])
            boxes.append([x1, y1, x2, y2])
            areas.append((y2 - y1) * (x2 - x1))
            iscrowd.append(False)
            labels.append(1)

        image = Image.fromarray(paper)
        image = self.image_transform(image)

        return (
            image,
            {
                "boxes": torch.FloatTensor(boxes),
                "labels": torch.LongTensor(labels),
                "image_id": torch.LongTensor([idx]),
                "area": torch.FloatTensor(areas),
                "iscrowd": torch.IntTensor(iscrowd),
            },
        )
