import shutil
import matplotlib
matplotlib.use("Agg")
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np


def _load_and_crop(path, img_size, crop, mode="RGB"):
    img = Image.open(path).convert(mode)
    if crop:
        w, h = img.size
        img = img.crop((crop.get("left", 0), crop.get("top", 0),
                        w - crop.get("right", 0), h - crop.get("bottom", 0)))
    return img.resize((img_size[1], img_size[0]))


def save_visualizations(anomaly_map, image_paths, ground_truth_dir, output_dir="output", img_size=(224, 224), crop=None):
    out_dir = Path(output_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    gt_root = Path(ground_truth_dir)

    maps_np = anomaly_map.detach().cpu().float()
    print(f"Anomaly score range: min={maps_np.min():.4f}, max={maps_np.max():.4f}, mean={maps_np.mean():.4f}")

    maps_up = F.interpolate(
        maps_np.unsqueeze(1),
        size=img_size,
        mode='bicubic',
        align_corners=False
    ).clamp(0, 1).squeeze(1).numpy()  # (B, H, W)

    colormap = plt.cm.jet

    for i, img_path in enumerate(image_paths):
        class_name = img_path.parent.name
        stem = img_path.stem

        original_np = np.array(_load_and_crop(img_path, img_size, crop, "RGB")) / 255.0

        gt_path = gt_root / class_name / f"{stem}_mask.png"
        if gt_path.exists():
            gt_img = np.array(_load_and_crop(gt_path, img_size, crop, "L"))
        else:
            gt_img = np.zeros(img_size, dtype=np.uint8)

        heat_pil = Image.fromarray((maps_up[i] * 255).astype(np.uint8))
        heat_smooth = np.array(heat_pil.filter(ImageFilter.GaussianBlur(radius=12))) / 255.0
        heat_colored = colormap(heat_smooth)[:, :, :3]
        overlay = 0.5 * original_np + 0.5 * heat_colored

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(original_np)
        axes[0].set_title("Originale")
        axes[0].axis("off")

        axes[1].imshow(gt_img, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(np.clip(overlay, 0, 1))
        axes[2].set_title("Anomaly Heatmap")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(out_dir / f"{class_name}_{stem}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"Visualizzazioni salvate in {out_dir}/ ({len(image_paths)} immagini)")
