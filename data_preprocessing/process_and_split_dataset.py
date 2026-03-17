'''
This script is used to process the raw dataset by
1) binarizing masks,
2) normalizing images between 0 and 1,
3) resizing to (256, 256),
4) splitting into train/val/test sub-datasets.
'''


import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def is_mask_file(Path):
    """
    Return True if the file is a mask file.
    This dataset usually uses '_mask' in the filename.
    """
    if not os.path.isfile(Path):
        return False

    file_name = os.path.basename(Path).lower()
    file_stem, _ = os.path.splitext(file_name)

    return "_mask" in file_stem


def is_image_file(Path):
    """
    Return True if the file is an image file but not a mask file.
    """
    if not os.path.isfile(Path):
        return False

    valid_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    _, ext = os.path.splitext(Path.lower())

    if ext not in valid_exts:
        return False

    return not is_mask_file(Path)


def get_mask_path_from_image_path(image_path):
    """
    Convert an image path to its corresponding mask path.
    Example:
    image: case_1_10.png
    mask : case_1_10_mask.png
    """
    folder = os.path.dirname(image_path)
    file_name = os.path.basename(image_path)
    file_stem, file_ext = os.path.splitext(file_name)

    mask_name = file_stem + "_mask" + file_ext
    mask_path = os.path.join(folder, mask_name)

    return mask_path


def collect_image_mask_pairs(dataset_path):
    """
    Recursively collect all valid image-mask pairs.
    Group them by patient folder for patient-level split.
    """
    if not os.path.isdir(dataset_path):
        raise ValueError(f"Invalid dataset folder: {dataset_path}")

    patient_dict = {}

    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            full_path = os.path.join(root, file_name)

            if is_image_file(full_path):
                mask_path = get_mask_path_from_image_path(full_path)

                if os.path.isfile(mask_path):
                    patient_id = os.path.basename(root)

                    if patient_id not in patient_dict:
                        patient_dict[patient_id] = []

                    patient_dict[patient_id].append((full_path, mask_path))

    return patient_dict


def normalize_image(Image):
    """
    Apply per-image min-max normalization and convert back to uint8 [0, 255].
    """
    image = Image.astype(np.float32)

    min_value = image.min()
    max_value = image.max()

    if max_value > min_value:
        image = (image - min_value) / (max_value - min_value)
    else:
        image = np.zeros_like(image, dtype=np.float32)

    image = (image * 255.0).clip(0, 255).astype(np.uint8)
    return image


def binarize_mask(Mask):
    """
    Convert mask to binary values: background 0, foreground 255.
    """
    binary_mask = np.where(Mask > 0, 255, 0).astype(np.uint8)
    return binary_mask


def preprocess_image_and_mask(image_path, mask_path, target_size):
    """
    Read image and mask with OpenCV, then apply:
    1. mask binarization
    2. image normalization
    3. resize
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_path}")

    image = normalize_image(image)
    mask = binarize_mask(mask)

    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    return image, mask


def create_output_folders(processed_dataset_path):
    """
    Create output folder structure:
    processed_dataset/
        train/
            images/
            masks/
        validation/
            images/
            masks/
        test/
            images/
            masks/
    """
    split_names = ["train", "validation", "test"]

    for split_name in split_names:
        image_folder = os.path.join(processed_dataset_path, split_name, "images")
        mask_folder = os.path.join(processed_dataset_path, split_name, "masks")

        os.makedirs(image_folder, exist_ok=True)
        os.makedirs(mask_folder, exist_ok=True)


def save_split_data(patient_dict, patient_ids, split_name, processed_dataset_path, target_size):
    """
    Preprocess and save all samples belonging to the given patient IDs.
    """
    image_output_folder = os.path.join(processed_dataset_path, split_name, "images")
    mask_output_folder = os.path.join(processed_dataset_path, split_name, "masks")

    saved_count = 0

    for patient_id in patient_ids:
        pairs = patient_dict[patient_id]

        for image_path, mask_path in pairs:
            image, mask = preprocess_image_and_mask(image_path, mask_path, target_size)

            original_name = os.path.basename(image_path)
            original_stem, original_ext = os.path.splitext(original_name)

            new_image_name = patient_id + "__" + original_stem + original_ext
            new_mask_name = patient_id + "__" + original_stem + "_mask" + original_ext

            image_save_path = os.path.join(image_output_folder, new_image_name)
            mask_save_path = os.path.join(mask_output_folder, new_mask_name)

            cv2.imwrite(image_save_path, image)
            cv2.imwrite(mask_save_path, mask)

            saved_count += 1

    return saved_count


def split_patients(patient_ids, train_ratio, val_ratio, test_ratio, random_seed):
    """
    Split patient IDs into train, validation, and test sets.
    """
    total_ratio = train_ratio + val_ratio + test_ratio

    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    train_ids, temp_ids = train_test_split(
        patient_ids,
        test_size=(1.0 - train_ratio),
        random_state=random_seed,
        shuffle=True
    )

    val_portion_in_temp = val_ratio / (val_ratio + test_ratio)

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=(1.0 - val_portion_in_temp),
        random_state=random_seed,
        shuffle=True
    )

    return train_ids, val_ids, test_ids


def print_split_summary(patient_dict, train_ids, val_ids, test_ids):
    """
    Print the number of patients and slices in each split.
    """
    train_slices = sum(len(patient_dict[patient_id]) for patient_id in train_ids)
    val_slices = sum(len(patient_dict[patient_id]) for patient_id in val_ids)
    test_slices = sum(len(patient_dict[patient_id]) for patient_id in test_ids)

    print("Split summary:")
    print(f"Train patients: {len(train_ids)}, Train slices: {train_slices}")
    print(f"Validation patients: {len(val_ids)}, Validation slices: {val_slices}")
    print(f"Test patients: {len(test_ids)}, Test slices: {test_slices}")


def main():
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    unprocessed_dataset = os.path.join(base_dir, "0_unprocessed_dataset/kaggle_3m")
    processed_dataset = os.path.join(base_dir, "1_processed_dataset")

    # Resize target: (width, height)
    target_size = (128, 128)

    # Split ratios
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15

    random_seed = 42

    patient_dict = collect_image_mask_pairs(unprocessed_dataset)

    if len(patient_dict) == 0:
        raise ValueError("No valid image-mask pairs were found in the dataset.")

    patient_ids = sorted(list(patient_dict.keys()))

    train_ids, val_ids, test_ids = split_patients(
        patient_ids,
        train_ratio,
        val_ratio,
        test_ratio,
        random_seed
    )

    create_output_folders(processed_dataset)

    train_count = save_split_data(patient_dict, train_ids, "train", processed_dataset, target_size)
    val_count = save_split_data(patient_dict, val_ids, "validation", processed_dataset, target_size)
    test_count = save_split_data(patient_dict, test_ids, "test", processed_dataset, target_size)

    print_split_summary(patient_dict, train_ids, val_ids, test_ids)
    print()
    print("Saved files summary:")
    print(f"Train samples saved: {train_count}")
    print(f"Validation samples saved: {val_count}")
    print(f"Test samples saved: {test_count}")
    print()
    print("Processing and splitting completed successfully.")


if __name__ == "__main__":
    main()