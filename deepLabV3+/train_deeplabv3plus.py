"""
This script trains DeepLabV3+ on 2_processed_dataset_with_data_augmentation.

Main settings:
- Model: DeepLabV3Plus
- Backbone: resnet34
- Pretrained weights: imagenet
- Input: 3-channel grayscale images
- Image size: 128 x 128
- Batch size: 16
- Epochs: up to 100
- Loss: BCEWithLogitsLoss + Dice Loss
- Optimizer: Adam
- Learning rate: 1e-4
- Scheduler: ReduceLROnPlateau
- Early stopping: patience = 8
- Best model criterion: highest validation Dice
- Mixed precision: enabled if CUDA is available
"""

import os
import csv
import copy
import time
import random
import warnings

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp


def set_seed(Seed):
    """
    Set random seed for reproducibility.
    """
    random.seed(Seed)
    np.random.seed(Seed)
    torch.manual_seed(Seed)
    torch.cuda.manual_seed_all(Seed)


def make_folder(Path):
    """
    Create the folder if it does not exist.
    """
    os.makedirs(Path, exist_ok=True)


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
    Example:
    TCGA_xxx__TCGA_xxx_10.tif
    ->
    TCGA_xxx__TCGA_xxx_10_mask.tif
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


def compute_batch_metrics(Logits, Targets, Threshold=0.5):
    """
    Compute batch-level Accuracy, Dice, and IoU.
    Metrics are averaged over samples in the batch.
    """
    preds = sigmoid_threshold(Logits, Threshold)
    targets = (Targets > 0.5).float()

    batch_size = preds.shape[0]

    accuracy_values = []
    dice_values = []
    iou_values = []

    for i in range(batch_size):
        pred_i = preds[i]
        target_i = targets[i]

        accuracy_values.append(compute_accuracy_from_tensors(pred_i, target_i))
        dice_values.append(compute_dice_from_tensors(pred_i, target_i))
        iou_values.append(compute_iou_from_tensors(pred_i, target_i))

    return (
        float(np.mean(accuracy_values)),
        float(np.mean(dice_values)),
        float(np.mean(iou_values)),
    )


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


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.
    Expects logits and binary targets.
    """

    def __init__(self, Epsilon=1e-7):
        super().__init__()
        self.epsilon = Epsilon

    def forward(self, Logits, Targets):
        probs = torch.sigmoid(Logits)

        probs = probs.contiguous().view(Logits.size(0), -1)
        targets = Targets.contiguous().view(Targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.epsilon) / (denominator + self.epsilon)
        loss = 1.0 - dice.mean()

        return loss


def compute_total_loss(Logits, Targets, BCELossFunction, DiceLossFunction):
    """
    Total loss = BCEWithLogitsLoss + DiceLoss
    """
    bce_loss = BCELossFunction(Logits, Targets)
    dice_loss = DiceLossFunction(Logits, Targets)
    total_loss = bce_loss + dice_loss
    return total_loss, bce_loss.item(), dice_loss.item()


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


def save_metrics_csv(HistoryRows, CsvPath):
    """
    Save training history to a CSV file.
    """
    if len(HistoryRows) == 0:
        return

    fieldnames = list(HistoryRows[0].keys())

    with open(CsvPath, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(HistoryRows)


def print_epoch_summary(Epoch, MaxEpochs, TrainLoss, TrainBCE, TrainDiceLoss, ValLoss, ValAccuracy, ValDice, ValIoU, CurrentLR):
    """
    Print a compact epoch summary.
    """
    print(
        f"Epoch [{Epoch}/{MaxEpochs}] | "
        f"Train Loss: {TrainLoss:.4f} | "
        f"Train BCE: {TrainBCE:.4f} | "
        f"Train Dice Loss: {TrainDiceLoss:.4f} | "
        f"Val Loss: {ValLoss:.4f} | "
        f"Val Acc: {ValAccuracy:.4f} | "
        f"Val Dice: {ValDice:.4f} | "
        f"Val IoU: {ValIoU:.4f} | "
        f"LR: {CurrentLR:.6f}"
    )


def train_one_epoch(Model, Loader, Optimizer, BCELossFunction, DiceLossFunction, Device, Scaler, UseAMP):
    """
    Train for one epoch.
    """
    Model.train()

    running_loss = 0.0
    running_bce = 0.0
    running_dice_loss = 0.0
    batch_count = 0

    for images, masks, _ in Loader:
        images = images.to(Device, non_blocking=True)
        masks = masks.to(Device, non_blocking=True)

        Optimizer.zero_grad(set_to_none=True)

        if UseAMP:
            with torch.amp.autocast(device_type="cuda"):
                logits = Model(images)
                total_loss, bce_loss_value, dice_loss_value = compute_total_loss(
                    logits, masks, BCELossFunction, DiceLossFunction
                )

            Scaler.scale(total_loss).backward()
            Scaler.step(Optimizer)
            Scaler.update()
        else:
            logits = Model(images)
            total_loss, bce_loss_value, dice_loss_value = compute_total_loss(
                logits, masks, BCELossFunction, DiceLossFunction
            )

            total_loss.backward()
            Optimizer.step()

        running_loss += total_loss.item()
        running_bce += bce_loss_value
        running_dice_loss += dice_loss_value
        batch_count += 1

    epoch_loss = running_loss / batch_count
    epoch_bce = running_bce / batch_count
    epoch_dice_loss = running_dice_loss / batch_count

    return epoch_loss, epoch_bce, epoch_dice_loss


@torch.no_grad()
def validate_one_epoch(Model, Loader, BCELossFunction, DiceLossFunction, Device, Threshold, UseAMP):
    """
    Run one validation epoch.
    """
    Model.eval()

    running_loss = 0.0
    running_accuracy = 0.0
    running_dice = 0.0
    running_iou = 0.0
    batch_count = 0

    for images, masks, _ in Loader:
        images = images.to(Device, non_blocking=True)
        masks = masks.to(Device, non_blocking=True)

        if UseAMP:
            with torch.amp.autocast(device_type="cuda"):
                logits = Model(images)
                total_loss, _, _ = compute_total_loss(
                    logits, masks, BCELossFunction, DiceLossFunction
                )
        else:
            logits = Model(images)
            total_loss, _, _ = compute_total_loss(
                logits, masks, BCELossFunction, DiceLossFunction
            )

        accuracy_value, dice_value, iou_value = compute_batch_metrics(
            logits, masks, Threshold
        )

        running_loss += total_loss.item()
        running_accuracy += accuracy_value
        running_dice += dice_value
        running_iou += iou_value
        batch_count += 1

    epoch_loss = running_loss / batch_count
    epoch_accuracy = running_accuracy / batch_count
    epoch_dice = running_dice / batch_count
    epoch_iou = running_iou / batch_count

    return epoch_loss, epoch_accuracy, epoch_dice, epoch_iou


def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    # -----------------------------
    # User settings
    # -----------------------------
    dataset_root = r"C:\Users\junhu\Desktop\CS4452 Group Project\2_processed_dataset_with_data_augmentation"
    results_dir = r"C:\Users\junhu\Desktop\CS4452 Group Project\deeplabv3plus_results\train_results"

    random_seed = 42
    batch_size = 16
    max_epochs = 100
    learning_rate = 1e-4
    num_workers = 4
    threshold = 0.5
    early_stopping_patience = 10

    # -----------------------------
    # Prepare folders and device
    # -----------------------------
    make_folder(results_dir)
    set_seed(random_seed)

    train_dir = os.path.join(dataset_root, "train")
    validation_dir = os.path.join(dataset_root, "validation")

    device = get_device()
    use_amp = torch.cuda.is_available()

    print(f"Device: {device}")
    print(f"Mixed precision enabled: {use_amp}")

    # -----------------------------
    # Datasets and loaders
    # -----------------------------
    train_dataset = BrainMRIDataset(train_dir)
    validation_dataset = BrainMRIDataset(validation_dir)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # -----------------------------
    # Model, loss, optimizer, scheduler
    # -----------------------------
    model = build_model().to(device)

    bce_loss_function = nn.BCEWithLogitsLoss()
    dice_loss_function = DiceLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        verbose=True
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # -----------------------------
    # Training state
    # -----------------------------
    history_rows = []
    best_val_dice = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    best_model_state = None

    start_time = time.time()

    # -----------------------------
    # Train / validate loop
    # -----------------------------
    for epoch in range(1, max_epochs + 1):
        train_loss, train_bce, train_dice_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            bce_loss_function,
            dice_loss_function,
            device,
            scaler,
            use_amp
        )

        val_loss, val_accuracy, val_dice, val_iou = validate_one_epoch(
            model,
            validation_loader,
            bce_loss_function,
            dice_loss_function,
            device,
            threshold,
            use_amp
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_dice)

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_bce_loss": train_bce,
            "train_dice_loss": train_dice_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "learning_rate": current_lr,
        })

        print_epoch_summary(
            epoch,
            max_epochs,
            train_loss,
            train_bce,
            train_dice_loss,
            val_loss,
            val_accuracy,
            val_dice,
            val_iou,
            current_lr
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            epochs_without_improvement = 0
            best_model_state = copy.deepcopy(model.state_dict())

            best_model_path = os.path.join(results_dir, "best_model.pth")
            torch.save(best_model_state, best_model_path)

            print(f"Best model updated at epoch {epoch} (Val Dice = {val_dice:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"No validation Dice improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    total_training_time = time.time() - start_time

    final_model_path = os.path.join(results_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)

    metrics_csv_path = os.path.join(results_dir, "training_history.csv")
    save_metrics_csv(history_rows, metrics_csv_path)

    summary_path = os.path.join(results_dir, "training_summary.txt")
    with open(summary_path, "w") as file:
        file.write(f"Best epoch: {best_epoch}\n")
        file.write(f"Best validation Dice: {best_val_dice:.6f}\n")
        file.write(f"Total training time (seconds): {total_training_time:.2f}\n")
        file.write(f"Device: {device}\n")
        file.write(f"Mixed precision enabled: {use_amp}\n")

    print()
    print("Training completed.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Total training time: {total_training_time / 60.0:.2f} minutes")
    print(f"Saved: {os.path.join(results_dir, 'best_model.pth')}")
    print(f"Saved: {final_model_path}")
    print(f"Saved: {metrics_csv_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()