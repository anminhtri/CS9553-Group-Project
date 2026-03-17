'''
This script is used to create a new dataset folder with data augmentation applied to the training set.
It keeps all original training samples, validation samples, and test samples, and only performs data
augmentation on training images whose corresponding masks are non-empty. The augmented samples are then
added to the training set, while the validation and test sets remain unchanged.
'''


import os
import cv2
import shutil
import random
import albumentations as A


def make_folder(Path):
    """
    Create the folder if it does not exist.
    """
    os.makedirs(Path, exist_ok=True)


def copy_split_folder(source_split_path, target_split_path):
    """
    Copy one split folder from source to target.
    """
    if os.path.exists(target_split_path):
        shutil.rmtree(target_split_path)

    shutil.copytree(source_split_path, target_split_path)


def get_image_files(images_folder):
    """
    Return all image file names in sorted order.
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
    Example:
    TCGA_xxx__TCGA_xxx_10.tif
    ->
    TCGA_xxx__TCGA_xxx_10_mask.tif
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


def binarize_mask(Mask):
    """
    Keep the mask strictly binary: 0 or 255.
    """
    binary_mask = (Mask > 0).astype("uint8") * 255
    return binary_mask


def is_non_empty_mask(Mask):
    """
    Return True if the mask contains any foreground pixel.
    """
    return Mask.max() > 0


def build_augmentation_pipeline():
    """
    Build the augmentation pipeline.
    """
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=15,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.7
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.5
        )
    ])

    return transform


def save_image_and_mask(image, mask, image_save_path, mask_save_path):
    """
    Save image and mask to disk.
    """
    cv2.imwrite(image_save_path, image)
    cv2.imwrite(mask_save_path, mask)


def augment_train_split(source_train_path, target_train_path, augmentations_per_image, random_seed):
    """
    Keep all original training data.
    Create augmented samples only for training images whose masks are non-empty.
    """
    random.seed(random_seed)

    source_images_folder = os.path.join(source_train_path, "images")
    source_masks_folder = os.path.join(source_train_path, "masks")

    target_images_folder = os.path.join(target_train_path, "images")
    target_masks_folder = os.path.join(target_train_path, "masks")

    image_files = get_image_files(source_images_folder)
    transform = build_augmentation_pipeline()

    original_count = 0
    eligible_non_empty_count = 0
    augmented_count = 0
    skipped_empty_count = 0

    for image_name in image_files:
        image_path = os.path.join(source_images_folder, image_name)
        mask_name = get_mask_name_from_image_name(image_name)
        mask_path = os.path.join(source_masks_folder, mask_name)

        if not os.path.isfile(mask_path):
            raise ValueError(f"Missing mask for image: {image_name}")

        image = read_image(image_path)
        mask = read_mask(mask_path)
        mask = binarize_mask(mask)

        original_count += 1

        if not is_non_empty_mask(mask):
            skipped_empty_count += 1
            continue

        eligible_non_empty_count += 1

        file_stem, file_ext = os.path.splitext(image_name)

        for augmentation_index in range(augmentations_per_image):
            transformed = transform(image=image, mask=mask)

            augmented_image = transformed["image"]
            augmented_mask = transformed["mask"]
            augmented_mask = binarize_mask(augmented_mask)

            augmented_image_name = file_stem + f"_aug_{augmentation_index + 1}" + file_ext
            augmented_mask_name = file_stem + f"_aug_{augmentation_index + 1}_mask" + file_ext

            augmented_image_path = os.path.join(target_images_folder, augmented_image_name)
            augmented_mask_path = os.path.join(target_masks_folder, augmented_mask_name)

            save_image_and_mask(
                augmented_image,
                augmented_mask,
                augmented_image_path,
                augmented_mask_path
            )

            augmented_count += 1

    return original_count, eligible_non_empty_count, skipped_empty_count, augmented_count


def count_files_in_folder(Path):
    """
    Count image files in a folder.
    """
    valid_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    count = 0

    for file_name in os.listdir(Path):
        full_path = os.path.join(Path, file_name)

        if os.path.isfile(full_path):
            _, ext = os.path.splitext(file_name.lower())
            if ext in valid_exts:
                count += 1

    return count


def main():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dataset = os.path.join(base_dir, "1_processed_dataset")
    processed_dataset_with_data_augmentation = os.path.join(base_dir, "2_processed_dataset_with_data_augmentation")

    # How many augmented samples to generate for each non-empty training image
    augmentations_per_image = 1

    random_seed = 42

    train_source = os.path.join(processed_dataset, "train")
    validation_source = os.path.join(processed_dataset, "validation")
    test_source = os.path.join(processed_dataset, "test")

    train_target = os.path.join(processed_dataset_with_data_augmentation, "train")
    validation_target = os.path.join(processed_dataset_with_data_augmentation, "validation")
    test_target = os.path.join(processed_dataset_with_data_augmentation, "test")

    if os.path.exists(processed_dataset_with_data_augmentation):
        shutil.rmtree(processed_dataset_with_data_augmentation)

    make_folder(processed_dataset_with_data_augmentation)

    # Copy original processed dataset first
    copy_split_folder(train_source, train_target)
    copy_split_folder(validation_source, validation_target)
    copy_split_folder(test_source, test_target)

    # Only augment non-empty samples in training split
    original_count, eligible_non_empty_count, skipped_empty_count, augmented_count = augment_train_split(
        train_source,
        train_target,
        augmentations_per_image,
        random_seed
    )

    final_train_images = count_files_in_folder(os.path.join(train_target, "images"))
    final_train_masks = count_files_in_folder(os.path.join(train_target, "masks"))
    final_validation_images = count_files_in_folder(os.path.join(validation_target, "images"))
    final_test_images = count_files_in_folder(os.path.join(test_target, "images"))

    print("Data augmentation completed successfully.")
    print()
    print(f"Original training samples kept: {original_count}")
    print(f"Non-empty training samples augmented: {eligible_non_empty_count}")
    print(f"Empty training samples skipped for augmentation: {skipped_empty_count}")
    print(f"New augmented training samples added: {augmented_count}")
    print()
    print(f"Final training images: {final_train_images}")
    print(f"Final training masks: {final_train_masks}")
    print(f"Validation images (unchanged): {final_validation_images}")
    print(f"Test images (unchanged): {final_test_images}")


if __name__ == "__main__":
    main()