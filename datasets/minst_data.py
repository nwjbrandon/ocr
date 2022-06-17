import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as datasets


def concat_image_horizontally(im1, im2, offset, threshold):
    h1, w1 = im1.shape
    h2, w2 = im2.shape

    # Render images horizontally
    im = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
    im[0:h1, 0:w1] = im1
    im[0:h2, w1 - offset : w1 - offset + w2] = im2

    # Render overlapping parts of images
    overlap1 = im1[0:h1, w1 - offset : w1].copy()
    overlap2 = im2[0:h2, 0:offset].copy()
    overlap1[overlap2 > threshold] = 255
    im[0:h1, w1 - offset : w1] = overlap1

    return im


def generate_data_of_n_characters(dataset, n_char=3, threshold=100):
    n1 = random.randint(0, len(dataset) - 1)
    img1 = np.asarray(dataset[n1][0]).copy()
    text = str(dataset[n1][1])
    positions = [
        (img1.shape[0] // 2, img1.shape[1] // 2),
    ]

    # Combine images of characters
    for _ in range(n_char):
        offset = random.randint(0, 6)
        n2 = random.randint(0, len(dataset))
        img2 = np.asarray(dataset[n2][0]).copy()
        text += str(dataset[n2][1])
        prev_h, prev_w = positions[-1]
        h, w = img2.shape
        new_h = prev_h
        new_w = prev_w + w - offset

        positions.append((new_h, new_w))

        img1 = concat_image_horizontally(img1, img2, offset, threshold)

    # Set written characters to black
    mask = img1 > threshold
    img1[mask] = 0
    img1[~mask] = 255

    return img1, text, positions


def visualize_centers(img, positions, heatmap=None, bnb_box=None):
    if heatmap is None:
        heatmap = np.zeros_like(img)

    for pos in positions:
        radius = 1
        color = 255
        thickness = -1
        heatmap = cv2.circle(
            heatmap, (pos[1], pos[0]), radius, color, thickness
        )

    if bnb_box is not None:
        color = 0
        thickness = 1
        cv2.rectangle(img, bnb_box[0], bnb_box[1], color, thickness)

    return heatmap


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


def render_data_on_paper(
    img,
    positions,
    angle=None,
    max_scale=None,
    offset=None,
    padding=1,
    paper=None,
    paper_size=(60, 100),
):
    if paper is None:
        paper = np.ones(paper_size, dtype=np.uint8) * 255
    paper_h, paper_w = paper.shape

    if angle is None:
        angle = random.randint(-15, 15)
    img, rotation_mat = rotate_image(img, angle)
    positions = rotate_points(positions, rotation_mat)

    h, w = img.shape
    if max_scale is None:
        max_scale = int(min(paper_h / h, paper_w / w) * 10) / 10
    scale = random.uniform(0.8, max_scale)
    img = resize_image(img, scale)
    positions = scale_points(positions, scale)

    h, w = img.shape
    if offset is None:
        offset_h = random.randint(padding, paper_h - h - padding)
        offset_w = random.randint(padding, paper_w - w - padding)
    else:
        offset_h = offset[0]
        offset_w = offset[1]
    paper[offset_h + 1 : offset_h + h + 1, offset_w : offset_w + w] = img
    positions = translate_points(positions, offset_h, offset_w)

    return paper, positions, (offset_w, offset_h), (offset_w + w, offset_h + h)


def render_multiple_data_on_paper(
    n_data, dataset, paper=None, max_scale=1.2, paper_size=(300, 300)
):
    if paper is None:
        paper = np.ones(paper_size, dtype=np.uint8) * 255
    paper_h, paper_w = paper.shape

    positions_list = list()
    bnb_box_list = list()
    text_list = list()

    mask = np.zeros_like(paper)
    for _ in range(n_data):
        n_char = random.randint(0, 2)
        img, text, positions = generate_data_of_n_characters(
            dataset, n_char=n_char
        )
        next_paper, positions, tl, br = render_data_on_paper(
            img, positions, max_scale=max_scale, paper=paper.copy()
        )
        bnb_box = (tl, br)

        local_mask = mask[tl[1] : br[1], tl[0] : br[0]]
        if np.any(local_mask):
            continue

        mask[tl[1] : br[1], tl[0] : br[0]] = 1
        positions_list.append(positions)
        bnb_box_list.append(bnb_box)
        text_list.append(text)
        paper = next_paper

    return paper, positions_list, bnb_box_list, text_list


def visualize_multiple_centers(paper, positions_list, bnb_box_list=None):
    n_data = len(positions_list)

    heatmap = np.zeros_like(paper)
    for i in range(n_data):
        positions = positions_list[i]
        bnb_box = None if bnb_box_list is None else bnb_box_list[i]
        heatmap = visualize_centers(
            paper, positions, heatmap=heatmap, bnb_box=bnb_box
        )
    return heatmap


def create_region_gt(paper, positions_list, k=5):
    heatmap = np.zeros_like(paper)
    for positions in positions_list:
        for pos in positions:
            heatmap[pos[0], pos[1]] = 255
    heatmap = cv2.GaussianBlur(heatmap, (k, k), 0)
    return heatmap


def create_affinity_gt(paper, positions_list, k=5):
    heatmap = np.zeros_like(paper)
    for positions in positions_list:
        for i in range(len(positions) - 1):
            pos = positions[i]
            if len(positions) == 1:
                continue

            pos_ = positions[i + 1]

            cy = (pos[0] + pos_[0]) // 2
            cx = (pos[1] + pos_[1]) // 2

            heatmap[cy, cx] = 255
    heatmap = cv2.GaussianBlur(heatmap, (k, k), 0)
    return heatmap


def test1():
    mnist_trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=None
    )
    img, text, positions = generate_data_of_n_characters(
        mnist_trainset, n_char=2
    )
    print(text)
    heatmap = visualize_centers(img, positions)
    _, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(img)
    ax[0].set_title("Image")
    ax[1].imshow(heatmap)
    ax[1].set_title("Centers")
    plt.show()


def test2():
    mnist_trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=None
    )
    img, text, positions = generate_data_of_n_characters(
        mnist_trainset, n_char=1
    )
    print(text)
    paper, positions, top_left, top_right = render_data_on_paper(img, positions)
    bnb_box = (top_left, top_right)
    heatmap = visualize_centers(paper, positions, bnb_box=bnb_box)
    _, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(paper)
    ax[0].set_title("Image")
    ax[1].imshow(heatmap)
    ax[1].set_title("Centers")
    plt.show()


def test3():
    mnist_trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=None
    )
    (
        paper,
        positions_list,
        bnb_box_list,
        text_list,
    ) = render_multiple_data_on_paper(20, mnist_trainset, paper_size=(300, 300))
    print("n_data:", len(positions_list))
    heatmap = visualize_multiple_centers(
        paper, positions_list, bnb_box_list=bnb_box_list
    )
    _, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(paper)
    ax[0].set_title("Image")
    ax[1].imshow(heatmap)
    ax[1].set_title("Centers")
    plt.show()


def test4():
    mnist_trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=None
    )
    (
        paper,
        positions_list,
        bnb_box_list,
        text_list,
    ) = render_multiple_data_on_paper(20, mnist_trainset, paper_size=(300, 300))
    region_gt = create_region_gt(paper, positions_list)
    affinity_gt = create_affinity_gt(paper, positions_list)
    print("n_data:", len(positions_list))
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


if __name__ == "__main__":
    test1()
    test2()
    test3()
    test4()
