"""
This script evaluates the saved best DeepLabV3+ model on the test set and
also plots the saved training curves from training_history.csv.

Main settings:
- Model: DeepLabV3Plus
- Backbone: resnet34
- Pretrained weights: imagenet
- Input: 3-channel grayscale images
- Best checkpoint: best_model.pth
- Final test metrics: Accuracy, Dice, IoU, HD
"""

import os
import csv
import math
import warnings

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff

import torch
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp


def get_image_files(ImagesFolder):
    """
    Return all valid image files in sorted order.
    """
    valid_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    image_files = []

    for file_name in os.listdir(ImagesFolder):
        full_path = os.path.join(ImagesFolder, file_name)

        if os.path.isfile(full_path):
            _, ext = os.path.splitext(file_name.lower())
            if ext in valid_exts:
                image_files.append(file_name)

    image_files.sort()
    return image_files


def get_mask_name_from_image_name(ImageName):
    """
    Convert image file name to its corresponding mask file name.
    """
    file_stem, file_ext = os.path.splitext(ImageName)
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


def prepare_image(Image):
    """
    Convert grayscale uint8 image to float32 in [0, 1],
    then duplicate it to 3 channels.
    """
    image = Image.astype(np.float32) / 255.0
    image = np.stack([image, image, image], axis=0)
    return image


def prepare_mask(Mask):
    """
    Convert mask from 0/255 to float32 0/1.
    """
    mask = (Mask > 0).astype(np.float32)
    mask = np.expand_dims(mask, axis=0)
    return mask


def sigmoid_threshold(Logits, Threshold=0.5):
    """
    Convert logits to binary predictions using sigmoid + threshold.
    """
    probs = torch.sigmoid(Logits)
    preds = (probs >= Threshold).float()
    return preds


def compute_accuracy_from_tensors(PredMask, TrueMask):
    """
    Compute pixel accuracy on tensors.
    Both inputs should already be binary {0,1}.
    """
    correct = (PredMask == TrueMask).float().sum()
    total = torch.numel(TrueMask)
    return (correct / total).item()


def compute_dice_from_tensors(PredMask, TrueMask, Epsilon=1e-7):
    """
    Compute Dice score on tensors.
    Both inputs should already be binary {0,1}.
    """
    intersection = (PredMask * TrueMask).sum()
    denominator = PredMask.sum() + TrueMask.sum()

    dice = (2.0 * intersection + Epsilon) / (denominator + Epsilon)
    return dice.item()


def compute_iou_from_tensors(PredMask, TrueMask, Epsilon=1e-7):
    """
    Compute IoU score on tensors.
    Both inputs should already be binary {0,1}.
    """
    intersection = (PredMask * TrueMask).sum()
    union = PredMask.sum() + TrueMask.sum() - intersection

    iou = (intersection + Epsilon) / (union + Epsilon)
    return iou.item()


def get_foreground_points(BinaryMask2D):
    """
    Return foreground pixel coordinates as N x 2 array.
    """
    points = np.argwhere(BinaryMask2D > 0)
    return points


def compute_hd_from_numpy(PredMask2D, TrueMask2D):
    """
    Compute symmetric Hausdorff Distance between two binary 2D masks.

    Handling:
    - if both masks are empty -> HD = 0
    - if one is empty and the other is not -> HD = image diagonal
    """
    pred_points = get_foreground_points(PredMask2D)
    true_points = get_foreground_points(TrueMask2D)

    if len(pred_points) == 0 and len(true_points) == 0:
        return 0.0

    height, width = PredMask2D.shape
    image_diagonal = math.sqrt(height ** 2 + width ** 2)

    if len(pred_points) == 0 or len(true_points) == 0:
        return float(image_diagonal)

    hd_forward = directed_hausdorff(pred_points, true_points)[0]
    hd_backward = directed_hausdorff(true_points, pred_points)[0]

    return float(max(hd_forward, hd_backward))


class BrainMRIDataset(Dataset):
    """
    Dataset for processed MRI segmentation data.
    """

    def __init__(self, SplitFolder):
        self.split_folder = SplitFolder
        self.images_folder = os.path.join(SplitFolder, "images")
        self.masks_folder = os.path.join(SplitFolder, "masks")

        if not os.path.isdir(self.images_folder):
            raise ValueError(f"Images folder not found: {self.images_folder}")

        if not os.path.isdir(self.masks_folder):
            raise ValueError(f"Masks folder not found: {self.masks_folder}")

        self.image_files = get_image_files(self.images_folder)

        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in: {self.images_folder}")

        for image_name in self.image_files:
            mask_name = get_mask_name_from_image_name(image_name)
            mask_path = os.path.join(self.masks_folder, mask_name)

            if not os.path.isfile(mask_path):
                raise ValueError(f"Missing mask for image: {image_name}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, Index):
        image_name = self.image_files[Index]
        mask_name = get_mask_name_from_image_name(image_name)

        image_path = os.path.join(self.images_folder, image_name)
        mask_path = os.path.join(self.masks_folder, mask_name)

        image = read_image(image_path)
        mask = read_mask(mask_path)

        image = prepare_image(image)
        mask = prepare_mask(mask)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        return image, mask, image_name


def get_device():
    """
    Automatically choose CUDA if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model():
    """
    Build DeepLabV3Plus model.
    """
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )
    return model


@torch.no_grad()
def evaluate_on_test(Model, Loader, Device, Threshold, UseAMP):
    """
    Final test evaluation.
    Computes Accuracy, Dice, IoU, and HD.
    """
    Model.eval()

    accuracy_values = []
    dice_values = []
    iou_values = []
    hd_values = []

    for images, masks, _ in Loader:
        images = images.to(Device, non_blocking=True)
        masks = masks.to(Device, non_blocking=True)

        if UseAMP:
            with torch.amp.autocast(device_type="cuda"):
                logits = Model(images)
        else:
            logits = Model(images)

        preds = sigmoid_threshold(logits, Threshold)
        targets = (masks > 0.5).float()

        batch_size = preds.shape[0]

        for i in range(batch_size):
            pred_i = preds[i, 0].detach().cpu().numpy().astype(np.uint8)
            target_i = targets[i, 0].detach().cpu().numpy().astype(np.uint8)

            pred_tensor = torch.from_numpy(pred_i).float()
            target_tensor = torch.from_numpy(target_i).float()

            accuracy_values.append(compute_accuracy_from_tensors(pred_tensor, target_tensor))
            dice_values.append(compute_dice_from_tensors(pred_tensor, target_tensor))
            iou_values.append(compute_iou_from_tensors(pred_tensor, target_tensor))
            hd_values.append(compute_hd_from_numpy(pred_i, target_i))

    results = {
        "test_accuracy": float(np.mean(accuracy_values)),
        "test_dice": float(np.mean(dice_values)),
        "test_iou": float(np.mean(iou_values)),
        "test_hd": float(np.mean(hd_values)),
    }

    return results


def save_test_results(TestResults, SavePath):
    """
    Save final test metrics to a txt file.
    """
    with open(SavePath, "w") as file:
        for key, value in TestResults.items():
            file.write(f"{key}: {value:.6f}\n")


def load_history_from_csv(CsvPath):
    """
    Load training history from CSV.
    """
    history_rows = []

    with open(CsvPath, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            history_rows.append({
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "train_bce_loss": float(row["train_bce_loss"]),
                "train_dice_loss": float(row["train_dice_loss"]),
                "val_loss": float(row["val_loss"]),
                "val_accuracy": float(row["val_accuracy"]),
                "val_dice": float(row["val_dice"]),
                "val_iou": float(row["val_iou"]),
                "learning_rate": float(row["learning_rate"]),
            })

    return history_rows


def save_curves(HistoryRows, OutputFolder):
    """
    Save loss and metric curves.
    """
    epochs = [row["epoch"] for row in HistoryRows]

    train_loss = [row["train_loss"] for row in HistoryRows]
    val_loss = [row["val_loss"] for row in HistoryRows]
    val_accuracy = [row["val_accuracy"] for row in HistoryRows]
    val_dice = [row["val_dice"] for row in HistoryRows]
    val_iou = [row["val_iou"] for row in HistoryRows]
    learning_rates = [row["learning_rate"] for row in HistoryRows]

    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OutputFolder, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.plot(epochs, val_dice, label="Validation Dice")
    plt.plot(epochs, val_iou, label="Validation IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Validation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OutputFolder, "validation_metrics_curve.png"))
    plt.close()

    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(epochs, learning_rates, label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OutputFolder, "learning_rate_curve.png"))
    plt.close()


def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    dataset_root = r"C:\Users\junhu\Desktop\CS4452 Group Project\2_processed_dataset_with_data_augmentation"
    results_dir = r"C:\Users\junhu\Desktop\CS4452 Group Project\deeplabv3plus_results"

    batch_size = 16
    num_workers = 4
    threshold = 0.5

    best_model_path = os.path.join(results_dir, "best_model.pth")
    training_history_path = os.path.join(results_dir, "training_history.csv")
    test_results_path = os.path.join(results_dir, "test_results.txt")
    evaluation_summary_path = os.path.join(results_dir, "evaluation_summary.txt")

    test_dir = os.path.join(dataset_root, "test")

    device = get_device()
    use_amp = torch.cuda.is_available()
    pin_memory = torch.cuda.is_available()

    print(f"Device: {device}")
    print(f"Mixed precision enabled: {use_amp}")

    test_dataset = BrainMRIDataset(test_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = build_model().to(device)

    if not os.path.isfile(best_model_path):
        raise ValueError(f"Best model not found: {best_model_path}")

    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)

    test_results = evaluate_on_test(
        model,
        test_loader,
        device,
        threshold,
        use_amp
    )

    save_test_results(test_results, test_results_path)

    if not os.path.isfile(training_history_path):
        raise ValueError(f"Training history CSV not found: {training_history_path}")

    history_rows = load_history_from_csv(training_history_path)
    save_curves(history_rows, results_dir)

    with open(evaluation_summary_path, "w") as file:
        file.write(f"Device: {device}\n")
        file.write(f"Mixed precision enabled: {use_amp}\n")
        for key, value in test_results.items():
            file.write(f"{key}: {value:.6f}\n")
        file.write(f"Saved: {test_results_path}\n")
        file.write(f"Saved: {os.path.join(results_dir, 'loss_curve.png')}\n")
        file.write(f"Saved: {os.path.join(results_dir, 'validation_metrics_curve.png')}\n")
        file.write(f"Saved: {os.path.join(results_dir, 'learning_rate_curve.png')}\n")

    print()
    print("Evaluation completed.")
    for key, value in test_results.items():
        print(f"{key}: {value:.4f}")
    print(f"Saved: {test_results_path}")
    print(f"Saved: {os.path.join(results_dir, 'loss_curve.png')}")
    print(f"Saved: {os.path.join(results_dir, 'validation_metrics_curve.png')}")
    print(f"Saved: {os.path.join(results_dir, 'learning_rate_curve.png')}")
    print(f"Saved: {evaluation_summary_path}")


if __name__ == "__main__":
    main()
