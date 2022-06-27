import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.character_data import (
    CharacterDataset,
    generate_image_of_n_characters,
    render_data_on_paper,
    label_mapping_char_to_int
)


class CharacterRecDataset(Dataset):
    def __init__(self, cfg, is_train):
        self.is_visualise = cfg["dataset"]["is_visualise"]
        self.n_char = cfg["dataset"]["n_char"]
        self.is_train = is_train

        if self.is_train:
            self.dataset = CharacterDataset()
            print("Size of train set", len(self.dataset))

        else:
            self.dataset = CharacterDataset()
            print("Size of val set", len(self.dataset))

        self.n_data = cfg["dataset"]["n_data"]
        self.image_transform = transforms.Compose([transforms.ToTensor(),])

    def __len__(self):
        return self.n_data

    def _create_label(self, text):
        label = [0 for _ in range(self.n_char)]
        for i, char in enumerate(text):
            label[i] = label_mapping_char_to_int[char]
        return np.array(label)

    def __getitem__(self, idx):
        img, text = generate_image_of_n_characters(self.dataset, n_char=9)

        # Create paper
        paper_size = (224, 400)
        paper = np.ones(paper_size, dtype=np.uint8) * 255

        # Create horizontal lines on paper
        spacing = np.random.randint(50, 100)
        line_color = np.random.randint(0, 100)
        line_thickness = np.random.randint(1, 2)
        line_slant = np.random.randint(-10, 10)
        for i in range(0, len(paper), spacing):
            paper = cv2.line(
                paper,
                (0, i + line_slant),
                (paper.shape[1], i - line_slant),
                line_color,
                line_thickness,
            )
        paper = cv2.blur(paper, (5, 5))

        paper_h, paper_w = paper.shape

        # Scale to max first
        padding = 0
        final_h, final_w = paper_h - padding, paper_w - padding
        max_scale = min(final_w / img.shape[1], final_h / img.shape[0])
        width = int(img.shape[1] * max_scale)
        height = int(img.shape[0] * max_scale)
        dsize = (width, height)
        img = cv2.resize(img, dsize)

        paper, top_left, bot_right = render_data_on_paper(img, paper)

        if self.is_train:
            # augmentation
            k = np.random.randint(1, 3) * 2 + 1
            n_iter = np.random.randint(1, 5)
            is_thicker_font = np.random.randint(0, 1)

            # Set different thickness
            if is_thicker_font:
                paper = cv2.erode(paper, (k, k), n_iter)
            else:
                paper = cv2.dilate(paper, (k, k), n_iter)

        if self.is_visualise:
            paper = cv2.rectangle(paper, top_left, bot_right, 0, 2)
            plt.figure(figsize=(10, 10))
            plt.imshow(paper)
            plt.title(text)
            plt.show()

        text_gt = self._create_label(text)
        image = Image.fromarray(paper)
        image_inp = self.image_transform(image)
        return {"image_inp": image_inp, "text_gt": text_gt}
