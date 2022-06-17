import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.minst_data import (
    create_affinity_gt,
    create_region_gt,
    render_multiple_data_on_paper,
    visualize_multiple_centers
)


class MinstDataset(Dataset):
    def __init__(self, cfg, is_train):
        self.n_data = cfg["dataset"]["n_data"]
        self.is_visualise = cfg["dataset"]["is_visualise"]
        self.img_training_size = cfg["dataset"]["img_training_size"]
        self.is_train = is_train

        if self.is_train:
            self.mnist_trainset = datasets.MNIST(
                root="./data", train=True, download=True, transform=None
            )
            print("Size of val set", len(self.mnist_trainset))
        else:
            self.mnist_trainset = datasets.MNIST(
                root="./data", train=False, download=True, transform=None
            )
            print("Size of val set", len(self.mnist_trainset))

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
