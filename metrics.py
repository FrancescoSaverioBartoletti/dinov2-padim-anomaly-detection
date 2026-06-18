import json
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score, recall_score, precision_score, f1_score,
)
from scipy.ndimage import label as ndlabel


def _load_gt_mask(gt_root, img_path, img_size, crop):
    gt_path = gt_root / img_path.parent.name / f"{img_path.stem}_mask.png"
    if gt_path.exists():
        img = Image.open(gt_path).convert("L")
        if crop:
            w, h = img.size
            img = img.crop((crop.get("left", 0), crop.get("top", 0),
                            w - crop.get("right", 0), h - crop.get("bottom", 0)))
        img = img.resize((img_size[1], img_size[0]), Image.NEAREST)
        return (np.array(img) > 0).astype(np.uint8)
    return np.zeros(img_size, dtype=np.uint8)


def _upsample(anomaly_map, img_size):
    return F.interpolate(
        anomaly_map.unsqueeze(1).float(),
        size=img_size, mode='bilinear', align_corners=False
    ).squeeze(1).numpy()


def _threshold_metrics(labels, scores, threshold):
    pred = (scores >= threshold).astype(int)
    return {
        "accuracy":  round(float(accuracy_score(labels, pred)), 4),
        "recall":    round(float(recall_score(labels, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(labels, pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(labels, pred, zero_division=0)), 4),
        "threshold": round(threshold, 6),
    }


def compute_aupro(gt_masks, pred_maps, fpr_limit=0.3, num_thresholds=100):
    thresholds = np.linspace(pred_maps.min(), pred_maps.max(), num_thresholds)
    non_gt = ~gt_masks.astype(bool)
    non_gt_total = non_gt.sum()

    pros, fprs = [], []
    for t in thresholds:
        pred_bin = pred_maps > t

        region_overlaps = []
        for gt, pred in zip(gt_masks, pred_bin):
            if gt.max() == 0:
                continue
            labeled, n = ndlabel(gt)
            for c in range(1, n + 1):
                comp = labeled == c
                region_overlaps.append(float((pred & comp).sum() / comp.sum()))

        pros.append(np.mean(region_overlaps) if region_overlaps else 0.0)
        fprs.append(float((pred_bin & non_gt).sum()) / non_gt_total if non_gt_total > 0 else 0.0)

    fprs, pros = np.array(fprs), np.array(pros)
    idx = np.argsort(fprs)
    fprs, pros = fprs[idx], pros[idx]

    mask = fprs <= fpr_limit
    if mask.sum() < 2:
        return 0.0
    return round(float(np.trapezoid(pros[mask], fprs[mask]) / fpr_limit), 4)


def metrics():
    with open("settings.YAML") as f:
        config = yaml.safe_load(f)

    img_size = (config["data"]["height"], config["data"]["width"])
    crop = config["data"].get("crop", {})
    gt_root = Path(config["data"]["ground_truth_dir"])

    results = torch.load("output/results.pt", map_location="cpu")
    anomaly_map = results["anomaly_map"]           # (N, H_patch, W_patch)
    image_paths = [Path(p) for p in results["image_paths"]]
    N = len(image_paths)

    print(f"Calcolo metriche su {N} immagini...")

    # ── GT ──────────────────────────────────────────────────────────────────
    img_labels = np.array([0 if p.parent.name == "good" else 1 for p in image_paths])
    gt_masks   = np.stack([_load_gt_mask(gt_root, p, img_size, crop) for p in image_paths])

    # ── SCORES ──────────────────────────────────────────────────────────────
    img_scores = anomaly_map.flatten(1).max(dim=1).values.numpy()   # (N,)
    pred_maps  = _upsample(anomaly_map, img_size)                    # (N, H, W)

    threshold = 0.7

    # ── IMAGE-LEVEL ──────────────────────────────────────────────────────────
    img_auroc = round(float(roc_auc_score(img_labels, img_scores)), 4)
    img_m     = _threshold_metrics(img_labels, img_scores, threshold)

    # ── PIXEL-LEVEL ──────────────────────────────────────────────────────────
    px_scores_all = pred_maps.flatten()
    px_labels_all = gt_masks.flatten()

    # Subsample to 2M pixels for AUROC (memory-safe at 1120x1120)
    rng = np.random.default_rng(42)
    n_sample = min(2_000_000, len(px_scores_all))
    idx = rng.choice(len(px_scores_all), size=n_sample, replace=False)
    px_scores_s = px_scores_all[idx]
    px_labels_s = px_labels_all[idx]

    print(f"  Calcolo pixel AUROC (subsample {n_sample//1_000_000}M px)...")
    px_auroc = round(float(roc_auc_score(px_labels_s, px_scores_s)), 4)
    px_m     = _threshold_metrics(px_labels_all, px_scores_all, threshold)

    print("  Calcolo AUPRO...")
    aupro = compute_aupro(gt_masks, pred_maps)

    # ── RISULTATI ────────────────────────────────────────────────────────────
    out = {
        "n_images": N,
        "image_level": {"auroc": img_auroc, **img_m},
        "pixel_level": {"auroc": px_auroc, "aupro": aupro, **px_m},
    }

    print("\n=== IMAGE-LEVEL ===")
    for k, v in out["image_level"].items():
        print(f"  {k:<12}: {v}")

    print("\n=== PIXEL-LEVEL ===")
    for k, v in out["pixel_level"].items():
        print(f"  {k:<12}: {v}")

    out_path = Path("output/metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nMetriche salvate in {out_path}")


if __name__ == "__main__":
    metrics()
