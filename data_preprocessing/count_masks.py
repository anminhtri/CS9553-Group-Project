'''
This script is used to count - total number of masks, number of empty masks, number of non-empty masks.
'''


import os
import cv2


def is_mask_file(Path):
    """
    Return True if the file looks like a mask file.
    This dataset usually uses '_mask' in the mask filename.
    """
    if not os.path.isfile(Path):
        return False

    file_name = os.path.basename(Path).lower()
    file_stem, _ = os.path.splitext(file_name)

    return "_mask" in file_stem


def is_empty_mask(Path):
    """
    Return True if the mask contains only background pixels.
    """
    mask = cv2.imread(Path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Failed to read image: {Path}")

    return mask.max() == 0


def count_empty_and_nonempty_masks(dataset_path):
    """
    Recursively scan the dataset folder and count empty/non-empty mask files.
    """
    if not os.path.isdir(dataset_path):
        raise ValueError(f"Invalid dataset folder: {dataset_path}")

    empty_count = 0
    nonempty_count = 0

    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            full_path = os.path.join(root, file_name)

            if is_mask_file(full_path):
                if is_empty_mask(full_path):
                    empty_count += 1
                else:
                    nonempty_count += 1

    total_count = empty_count + nonempty_count
    return empty_count, nonempty_count, total_count


def main():
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    unprocessed_dataset = os.path.join(base_dir, "unprocessed_dataset/kaggle_3m")

    empty_count, nonempty_count, total_count = count_empty_and_nonempty_masks(unprocessed_dataset)

    print(f"Total mask files: {total_count}")    
    print(f"Empty masks: {empty_count}")
    print(f"Non-empty masks: {nonempty_count}")


if __name__ == "__main__":
    main()