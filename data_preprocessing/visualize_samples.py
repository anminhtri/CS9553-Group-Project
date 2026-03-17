"""
This script is used to visualize a few sample image-mask pairs from the final dataset.
It creates one high-resolution comparison figure for report use. Each sample set includes
the original MRI image, the binary mask, and an overlay image. The script prefers samples
with non-empty masks so that the tumor region can be clearly shown in the final figure.
"""

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


def get_image_files(images_folder):
    """
    Return all valid image files in sorted order.
    """
    valid_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    image_files = []

    for file_name in os.listdir(images_folder):
        full_path = os.path.join(images_folder, file_name)

        if os.path.isfile(full_path):
            _, ext = os.path.splitext(file_name.lower())
            if ext in valid_exts:
                image_files.append(file_name)

    image_files.sort()
    return image_files


def get_mask_name_from_image_name(image_name):
    """
    Convert image file name to its corresponding mask file name.
    """
    file_stem, file_ext = os.path.splitext(image_name)
    mask_name = file_stem + "_mask" + file_ext
    return mask_name


def read_image(Path):
    """
    Read a grayscale image with OpenCV.
    """
    image = cv2.imread(Path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Failed to read image: {Path}")

    return image


def read_mask(Path):
    """
    Read a grayscale mask with OpenCV.
    """
    mask = cv2.imread(Path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Failed to read mask: {Path}")

    return mask


def is_non_empty_mask(Mask):
    """
    Return True if the mask contains foreground.
    """
    return Mask.max() > 0


def collect_non_empty_samples(dataset_path):
    """
    Collect all non-empty samples from train, validation, and test.
    Each sample is stored as:
    (split_name, image_path, mask_path, image_name)
    """
    samples = []
    split_names = ["train", "validation", "test"]

    for split_name in split_names:
        images_folder = os.path.join(dataset_path, split_name, "images")
        masks_folder = os.path.join(dataset_path, split_name, "masks")

        image_files = get_image_files(images_folder)

        for image_name in image_files:
            image_path = os.path.join(images_folder, image_name)
            mask_name = get_mask_name_from_image_name(image_name)
            mask_path = os.path.join(masks_folder, mask_name)

            if not os.path.isfile(mask_path):
                continue

            mask = read_mask(mask_path)

            if is_non_empty_mask(mask):
                samples.append((split_name, image_path, mask_path, image_name))

    return samples


def choose_samples(samples, num_samples, random_seed):
    """
    Randomly choose a few samples for visualization.
    """
    random.seed(random_seed)

    if len(samples) < num_samples:
        raise ValueError(f"Not enough non-empty samples to choose {num_samples} examples.")

    selected_samples = random.sample(samples, num_samples)
    return selected_samples


def create_overlay(Image, Mask):
    """
    Create a red overlay for the mask region on top of the grayscale image.
    """
    image_rgb = cv2.cvtColor(Image, cv2.COLOR_GRAY2RGB)
    overlay = image_rgb.copy()

    red_mask = np.zeros_like(image_rgb)
    red_mask[:, :, 0] = 255

    mask_binary = (Mask > 0).astype(np.uint8)

    alpha = 0.4
    overlay[mask_binary == 1] = (
        (1 - alpha) * image_rgb[mask_binary == 1] + alpha * red_mask[mask_binary == 1]
    ).astype(np.uint8)

    return overlay


def make_visualization_figure(selected_samples, output_path):
    """
    Create one figure with 3 rows and 3 columns:
    image, mask, overlay for each selected sample.
    """
    num_samples = len(selected_samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(7.0, 7.2), dpi=300)

    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    column_titles = ["MRI Image", "Mask", "Overlay"]

    for col_index in range(3):
        axes[0, col_index].set_title(column_titles[col_index], fontsize=9)

    for row_index, sample in enumerate(selected_samples):
        split_name, image_path, mask_path, image_name = sample

        image = read_image(image_path)
        mask = read_mask(mask_path)
        overlay = create_overlay(image, mask)

        axes[row_index, 0].imshow(image, cmap="gray")
        axes[row_index, 1].imshow(mask, cmap="gray")
        axes[row_index, 2].imshow(overlay)

        short_name = os.path.splitext(image_name)[0]
        row_label = f"{split_name}: {short_name}"

        axes[row_index, 0].set_ylabel(row_label, fontsize=8)

        for col_index in range(3):
            axes[row_index, col_index].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def main():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dataset_with_data_augmentation = os.path.join(base_dir, "2_processed_dataset_with_data_augmentation")
    output_figure = os.path.join(base_dir, "sample_visualization.png")

    num_samples = 3
    random_seed = 42

    samples = collect_non_empty_samples(processed_dataset_with_data_augmentation)
    selected_samples = choose_samples(samples, num_samples, random_seed)
    make_visualization_figure(selected_samples, output_figure)

    print("Sample visualization completed successfully.")
    print(f"Saved figure: {output_figure}")


if __name__ == "__main__":
    main()