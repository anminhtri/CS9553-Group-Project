import os

# Set before importing torch / numpy / cv2 / matplotlib
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


def is_mask_non_empty(mask):
    return np.any(mask > 0)


def load_grayscale_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return image


def prepare_input_tensor_1ch(image):
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)   # (1, H, W)
    image_tensor = torch.from_numpy(image).float().unsqueeze(0)   # (1, 1, H, W)
    return image_tensor


def prepare_input_tensor_3ch(image):
    image = image.astype(np.float32) / 255.0
    image_3ch = np.stack([image, image, image], axis=0)   # (3, H, W)
    image_tensor = torch.from_numpy(image_3ch).float().unsqueeze(0)   # (1, 3, H, W)
    return image_tensor


def predict_mask_1ch(model, image, device, threshold=0.5):
    image_tensor = prepare_input_tensor_1ch(image).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        pred_mask = (probs > threshold).float()

    pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8)
    return pred_mask


def predict_mask_3ch(model, image, device, threshold=0.5):
    image_tensor = prepare_input_tensor_3ch(image).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        pred_mask = (probs > threshold).float()

    pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8)
    return pred_mask


def find_non_empty_test_samples(test_images_dir, test_masks_dir):
    image_paths = sorted(list(test_images_dir.glob("*")))
    mask_paths = sorted(list(test_masks_dir.glob("*")))

    mask_map = {}

    for mask_path in mask_paths:
        stem = mask_path.stem

        if stem.endswith("_mask"):
            key = stem[:-5]
        else:
            key = stem

        mask_map[key] = mask_path

    valid_samples = []

    for image_path in image_paths:
        image_key = image_path.stem

        if image_key not in mask_map:
            print(f"Warning: no matching mask found for image: {image_path.name}")
            continue

        mask_path = mask_map[image_key]
        mask = load_grayscale_image(mask_path)

        if is_mask_non_empty(mask):
            valid_samples.append((image_path, mask_path))

    return valid_samples


class DoubleConvUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvUNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        self.down1 = DoubleConvUNet(in_channels, 64)
        self.down2 = DoubleConvUNet(64, 128)
        self.down3 = DoubleConvUNet(128, 256)
        self.down4 = DoubleConvUNet(256, 512)

        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = DoubleConvUNet(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConvUNet(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = DoubleConvUNet(128, 64)

        self.outc = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(F.max_pool2d(x1, 2))
        x3 = self.down3(F.max_pool2d(x2, 2))
        x4 = self.down4(F.max_pool2d(x3, 2))

        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)

        logits = self.outc(x)
        return logits


class DoubleConvAttention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape != x1.shape:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.enc1 = DoubleConvAttention(in_channels, features[0])
        self.enc2 = DoubleConvAttention(features[0], features[1])
        self.enc3 = DoubleConvAttention(features[1], features[2])
        self.enc4 = DoubleConvAttention(features[2], features[3])

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConvAttention(features[3], features[3] * 2)

        self.up4 = nn.ConvTranspose2d(features[3] * 2, features[3], 2, stride=2)
        self.up3 = nn.ConvTranspose2d(features[3], features[2], 2, stride=2)
        self.up2 = nn.ConvTranspose2d(features[2], features[1], 2, stride=2)
        self.up1 = nn.ConvTranspose2d(features[1], features[0], 2, stride=2)

        self.att4 = AttentionGate(F_g=features[3], F_l=features[3], F_int=features[2])
        self.att3 = AttentionGate(F_g=features[2], F_l=features[2], F_int=features[1])
        self.att2 = AttentionGate(F_g=features[1], F_l=features[1], F_int=features[0])
        self.att1 = AttentionGate(F_g=features[0], F_l=features[0], F_int=features[0] // 2)

        self.dec4 = DoubleConvAttention(features[3] * 2, features[3])
        self.dec3 = DoubleConvAttention(features[2] * 2, features[2])
        self.dec2 = DoubleConvAttention(features[1] * 2, features[1])
        self.dec1 = DoubleConvAttention(features[0] * 2, features[0])

        self.out_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        b = self.bottleneck(self.pool(s4))

        d4 = self.up4(b)
        s4 = self.att4(g=d4, x=s4)
        d4 = self.dec4(torch.cat([d4, s4], dim=1))

        d3 = self.up3(d4)
        s3 = self.att3(g=d3, x=s3)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))

        d2 = self.up2(d3)
        s2 = self.att2(g=d2, x=s2)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))

        d1 = self.up1(d2)
        s1 = self.att1(g=d1, x=s1)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))

        return self.out_conv(d1)


def build_unet():
    model = UNet(in_channels=1, out_channels=1)
    return model


def build_attention_unet():
    model = AttentionUNet(in_channels=1, out_channels=1, features=[64, 128, 256, 512])
    return model


def build_deeplabv3plus():
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    return model


def load_model(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    trained_models_dir = Path(base_dir) / "trained_models"
    output_dir = Path(base_dir) / "plot_results"
    output_path = output_dir / "prediction_examples_3models_5cases.png"

    unet_model_path = trained_models_dir / "best_unet.pth"
    attention_unet_model_path = trained_models_dir / "best_attention_unet.pth"
    deeplab_model_path = trained_models_dir / "best_deeplabv3Plus_model.pth"

    test_images_dir = Path("/Users/zongjunhui/Desktop/western/2026 Winter/COMPSCI 9553/Group Project/code_n_data/2_processed_dataset_with_data_augmentation/test/images")
    test_masks_dir = Path("/Users/zongjunhui/Desktop/western/2026 Winter/COMPSCI 9553/Group Project/code_n_data/2_processed_dataset_with_data_augmentation/test/masks")

    num_examples = 5
    threshold = 0.5
    random_seed = 42

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"U-Net model path: {unet_model_path}")
    print(f"Attention U-Net model path: {attention_unet_model_path}")
    print(f"DeepLabV3+ model path: {deeplab_model_path}")
    print(f"Test images dir: {test_images_dir}")
    print(f"Test masks dir: {test_masks_dir}")

    unet_model = load_model(build_unet(), unet_model_path, device)
    attention_unet_model = load_model(build_attention_unet(), attention_unet_model_path, device)
    deeplab_model = load_model(build_deeplabv3plus(), deeplab_model_path, device)

    valid_samples = find_non_empty_test_samples(test_images_dir, test_masks_dir)
    print(f"Found {len(valid_samples)} non-empty matched samples")

    if len(valid_samples) == 0:
        raise ValueError("No non-empty ground truth masks found in the test set.")

    if len(valid_samples) < num_examples:
        print(f"Warning: only {len(valid_samples)} non-empty test samples found.")
        selected_samples = valid_samples
    else:
        random.seed(random_seed)
        selected_samples = random.sample(valid_samples, num_examples)

    num_rows = len(selected_samples)
    fig = plt.figure(figsize=(20, 4 * num_rows))
    gs = fig.add_gridspec(
        num_rows, 7,
        width_ratios=[1, 0.18, 1, 1, 1, 0.18, 1],
        wspace=0.02
    )

    axes = np.empty((num_rows, 5), dtype=object)
    for r in range(num_rows):
        axes[r, 0] = fig.add_subplot(gs[r, 0])  # Original
        axes[r, 1] = fig.add_subplot(gs[r, 2])  # U-Net
        axes[r, 2] = fig.add_subplot(gs[r, 3])  # Attention U-Net
        axes[r, 3] = fig.add_subplot(gs[r, 4])  # DeepLabV3+
        axes[r, 4] = fig.add_subplot(gs[r, 6])  # GT

    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    column_titles = ["Original Image", "U-Net", "Attention U-Net", "DeepLabV3+", "Ground Truth Mask"]
    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].set_title(title, fontsize=20, fontweight="bold", pad=12)

    for row_idx, (image_path, mask_path) in enumerate(selected_samples):
        image = load_grayscale_image(image_path)
        gt_mask = load_grayscale_image(mask_path)
        gt_mask_binary = (gt_mask > 0).astype(np.uint8)

        pred_unet = predict_mask_1ch(unet_model, image, device, threshold=threshold)
        pred_attention_unet = predict_mask_1ch(attention_unet_model, image, device, threshold=threshold)
        pred_deeplab = predict_mask_3ch(deeplab_model, image, device, threshold=threshold)

        axes[row_idx, 0].imshow(image, cmap="gray")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(pred_unet, cmap="gray")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(pred_attention_unet, cmap="gray")
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(pred_deeplab, cmap="gray")
        axes[row_idx, 3].axis("off")

        axes[row_idx, 4].imshow(gt_mask_binary, cmap="gray")
        axes[row_idx, 4].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()