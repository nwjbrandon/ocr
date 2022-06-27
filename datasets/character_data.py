import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

label_mapping_int_to_char = {
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "0",
    11: "a",
    12: "b",
    13: "c",
    14: "d",
    15: "e",
    16: "f",
    17: "g",
    18: "h",
    19: "i",
    20: "j",
    21: "v",
    22: "(",
    23: ")",
}
label_mapping_char_to_int = {
    label_mapping_int_to_char[key]: key for key in label_mapping_int_to_char
}


class CharacterDataset(Dataset):
    def __init__(self):
        self.dataset = [
            "my_data/nwjbrandon_20220701_big_1.jpg",
            "my_data/nwjbrandon_20220701_big_2.jpg",
            "my_data/nwjbrandon_20220701_big_3.jpg",
            "my_data/nwjbrandon_20220701_big_4.jpg",
            "my_data/nwjbrandon_20220701_big_5.jpg",
            "my_data/nwjbrandon_20220701_big_6.jpg",
            "my_data/nwjbrandon_20220701_big_7.jpg",
            "my_data/nwjbrandon_20220701_big_8.jpg",
            "my_data/nwjbrandon_20220701_big_9.jpg",
            "my_data/nwjbrandon_20220701_big_0.jpg",
            "my_data/nwjbrandon_20220701_big_a.jpg",
            "my_data/nwjbrandon_20220701_big_b.jpg",
            "my_data/nwjbrandon_20220701_big_c.jpg",
            "my_data/nwjbrandon_20220701_big_d.jpg",
            "my_data/nwjbrandon_20220701_big_e.jpg",
            "my_data/nwjbrandon_20220701_big_f.jpg",
            "my_data/nwjbrandon_20220701_big_g.jpg",
            "my_data/nwjbrandon_20220701_big_h.jpg",
            "my_data/nwjbrandon_20220701_big_i.jpg",
            "my_data/nwjbrandon_20220701_big_j.jpg",
            "my_data/nwjbrandon_20220701_big_v.jpg",
            "my_data/nwjbrandon_20220701_big_(.jpg",
            "my_data/nwjbrandon_20220701_big_).jpg",
        ]
        self.n_data = len(self.dataset)

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        img_fname = self.dataset[idx]
        image = Image.open(img_fname).convert("L")
        image = np.uint8(image)

        img_fname = img_fname.split(".")[0]
        label = img_fname.split("_")[-1]
        label = label_mapping_char_to_int[label]
        return image, label


def concat_image_horizontally(im1, im2, offset):
    h1, w1 = im1.shape
    h2, w2 = im2.shape

    # Render images horizontally
    im = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
    im[0:h1, 0:w1] = im1
    im[0:h2, w1 - offset : w1 - offset + w2] = im2

    # Render overlapping parts of images
    overlap1 = im1[0:h1, w1 - offset : w1].copy()
    overlap2 = im2[0:h2, 0:offset].copy()
    overlap = np.minimum(overlap1, overlap2)

    im[0:h1, w1 - offset : w1] = overlap
    return im[:, : w1 + w2 - offset]


def generate_image_of_n_characters(dataset, n_char=3):
    n1 = np.random.randint(0, len(dataset) - 1)
    img1 = dataset[n1][0]
    text = label_mapping_int_to_char[dataset[n1][1]]

    # Combine images of characters
    for _ in range(n_char):
        offset = np.random.randint(0, 100)
        offset = 100
        n2 = np.random.randint(0, len(dataset) - 1)
        img2 = dataset[n2][0]
        text += label_mapping_int_to_char[dataset[n2][1]]
        img1 = concat_image_horizontally(img1, img2, offset)

    img = img1
    padding = np.random.randint(5, 20)
    # Create tightly fitted bounding box
    _, bw_img = cv2.threshold(255 - img, 100, 255, cv2.THRESH_BINARY)
    x, y = np.where(bw_img == 255)
    top, bot = np.min(x), np.max(x)
    left, right = np.min(y), np.max(y)

    top = max(0, top - padding)
    left = max(0, left - padding)
    bot = min(img.shape[0], bot + padding)
    right = min(img.shape[1], right + padding)

    img = img[top:bot, left:right]

    return img, text


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
        width / 2,
        height / 2,
    )  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(
        mat, rotation_mat, (bound_w, bound_h), borderValue=255
    )
    return rotated_mat, rotation_mat


def resize_image(img, scale):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def rotate_points(points, rotation_mat):
    R = np.eye(3)
    R[0:2, 0:3] = rotation_mat
    points = [R @ np.array([pt[1], pt[0], 1]) for pt in points]
    points = [(int(pt[1]), int(pt[0])) for pt in points]
    return points


def translate_points(points, offset_h, offset_w):
    return [(pt[0] + offset_h, pt[1] + offset_w) for pt in points]


def scale_points(points, scale):
    return [(int(pt[0] * scale), int(pt[1] * scale)) for pt in points]


def render_data_on_paper(img, paper):
    paper_h, paper_w = paper.shape

    # Rotate image
    angle = np.random.randint(-5, 5)
    img, rotation_mat = rotate_image(img, angle)

    # Scale image
    scale = np.random.uniform(0.8, 3)
    img = resize_image(img, scale)

    # Translate image
    h, w = img.shape[:2]
    if w >= paper_w:
        scale = (paper_w - 1) / w
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dsize = (width, height)
        img = cv2.resize(img, dsize)
    if h >= paper_h:
        scale = (paper_h - 1) / h
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dsize = (width, height)
        img = cv2.resize(img, dsize)

    h, w = img.shape[:2]
    offset_h = np.random.randint(0, paper_h - h)
    offset_w = np.random.randint(0, paper_w - w)

    overlap = np.minimum(
        paper[offset_h : offset_h + h, offset_w : offset_w + w], img
    )
    paper[offset_h : offset_h + h, offset_w : offset_w + w] = overlap
    return paper, (offset_w, offset_h), (offset_w + w, offset_h + h)


def render_multiple_data_on_paper(dataset, n_data, paper):
    paper_h, paper_w = paper.shape

    bnb_box_list = list()
    text_list = list()

    mask = np.zeros_like(paper)
    for _ in range(n_data):
        n_char = np.random.randint(0, 9)
        img, text = generate_image_of_n_characters(dataset, n_char=n_char)

        max_scale = np.random.uniform(0.3, 0.5)
        width = int(img.shape[1] * max_scale)
        height = int(img.shape[0] * max_scale)
        dsize = (width, height)
        img = cv2.resize(img, dsize)

        next_paper, tl, br = render_data_on_paper(img, paper.copy())
        bnb_box = (tl, br)

        local_mask = mask[tl[1] : br[1], tl[0] : br[0]]
        if np.any(local_mask):
            continue

        mask[tl[1] : br[1], tl[0] : br[0]] = 1
        bnb_box_list.append(bnb_box)
        text_list.append(text)
        paper = next_paper

    return paper, bnb_box_list, text_list
