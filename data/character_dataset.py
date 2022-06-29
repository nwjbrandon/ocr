import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.character_data import (
    CharacterDataset,
    generate_image_of_n_characters,
    label_mapping_char_to_int,
    render_data_on_paper
)


class CharacterRecDataset(Dataset):
    def __init__(self, cfg, is_train):
        self.is_visualise = cfg["dataset"]["is_visualise"]
        self.n_char = cfg["dataset"]["n_char"]
        self.is_train = is_train

        if self.is_train:
            self.dataset = CharacterDataset(cfg["dataset"]["train_dir"])
            print("Size of train set", len(self.dataset))

        else:
            self.dataset = CharacterDataset(cfg["dataset"]["val_dir"])
            print("Size of val set", len(self.dataset))

        self.n_data = cfg["dataset"]["n_data"]
        self.image_transform = transforms.Compose([transforms.ToTensor(),])

    def __len__(self):
        # return self.n_data
        return 60000 if self.is_train else 10000

    def _create_label(self, text):
        label = [0 for _ in range(self.n_char)]
        for i, char in enumerate(text):
            label[i] = label_mapping_char_to_int[char]
        return np.array(label)

    def __getitem__(self, idx):
        n_char = np.random.randint(0, self.n_char)
        img, text = generate_image_of_n_characters(self.dataset, n_char=n_char)

        # Create paper
        paper_size = (128, 400)
        paper = np.ones(paper_size, dtype=np.uint8) * 255
        paper, top_left, bot_right = render_data_on_paper(img, paper)
        text_gt = self._create_label(text)
        _, paper = cv2.threshold(paper, 220, 255, cv2.THRESH_BINARY)
        paper = cv2.resize(paper, (200, 64))
        if self.is_visualise:
            print(
                "text:",
                text,
                "|",
                "n_char:",
                len(text),
                "|",
                "encoded:",
                text_gt,
            )
            # paper = cv2.rectangle(paper, top_left, bot_right, 0, 1)
            plt.figure(figsize=(10, 10))
            plt.imshow(paper)
            plt.title(text)
            plt.show()

        image = Image.fromarray(paper)
        image_inp = self.image_transform(image)
        return {
            "image_inp": image_inp,
            "text_gt": text_gt,
            "text_length": len(text),
        }
