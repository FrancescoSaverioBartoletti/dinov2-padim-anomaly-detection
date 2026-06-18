import time
import random
import yaml
import torchvision.transforms as transforms
from src.dataset import ToothbrushImageDataset, CropSides
import torch
from torch.utils.data import DataLoader, Subset
from src.dino_extractor import load_model, extract_dino_features
from src.padim import calculate_mahalanobis_distance, sigmoid_normalize
from src.utils import save_visualizations

def test(n_images: int = None, seed: int = None):
    with open("settings.YAML", "r") as f:
        config = yaml.safe_load(f)

    test_dir = config["data"]["test_dir"]
    ground_truth_dir = config["data"]["ground_truth_dir"]
    batch_size = config["data"]["batch_size"]
    height = config["data"]["height"]
    width = config["data"]["width"]
    k = config["inference"]["k"]
    crop = config["data"].get("crop", {})

    crop_transform = [CropSides(**crop)] if crop else []
    test_transform = transforms.Compose(crop_transform + [
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = ToothbrushImageDataset(test_dir, test_transform)

    if n_images is not None:
        rng = random.Random(seed)
        all_indices = list(range(len(test_dataset)))
        indices = rng.sample(all_indices, min(n_images, len(all_indices)))
        indices.sort()
        test_dataset = Subset(test_dataset, indices)
        image_paths = [test_dataset.dataset.paths[i] for i in indices]
        classes = [p.parent.name for p in image_paths]
        print(f"Subset casuale: {len(indices)} immagini — {dict((c, classes.count(c)) for c in set(classes))}")
    else:
        image_paths = test_dataset.paths

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    backbone = config["model"]["backbone"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(backbone, device)

    stats = torch.load("model/padim_stats.pt", map_location=device)
    mean = stats["mean"]
    inv_cov = stats["inv_cov"]
    tau_matrix = stats["tau_matrix"]
    selected_channels = stats["selected_channels"]

    t_start = time.perf_counter()
    features = extract_dino_features(model, test_dataloader, device)
    t_features = time.perf_counter()

    features_reduced = features[:, selected_channels.cpu(), :, :].to(device)
    distances = calculate_mahalanobis_distance(features_reduced, mean, inv_cov)
    t_distances = time.perf_counter()

    anomaly_map = sigmoid_normalize(distances, tau_matrix, k=k)
    t_sigmoid = time.perf_counter()

    n_imgs = len(image_paths)
    print(f"Distances: min={distances.min():.2f}, max={distances.max():.2f}, mean={distances.mean():.2f}")
    print(f"Tau matrix: min={tau_matrix.min():.2f}, max={tau_matrix.max():.2f}, mean={tau_matrix.mean():.2f}")
    print(f"Score range: min={anomaly_map.min():.4f}, max={anomaly_map.max():.4f}")
    print(f"--- Tempi di inferenza ({n_imgs} immagini) ---")
    print(f"  Feature extraction : {t_features - t_start:.2f}s  ({(t_features - t_start)/n_imgs*1000:.1f} ms/img)")
    print(f"  Mahalanobis dist.  : {t_distances - t_features:.2f}s  ({(t_distances - t_features)/n_imgs*1000:.1f} ms/img)")
    print(f"  Sigmoid normalize  : {t_sigmoid - t_distances:.4f}s  ({(t_sigmoid - t_distances)/n_imgs*1000:.2f} ms/img)")
    print(f"  Totale             : {t_sigmoid - t_start:.2f}s  ({(t_sigmoid - t_start)/n_imgs*1000:.1f} ms/img)")

    save_visualizations(anomaly_map, image_paths, ground_truth_dir, img_size=(height, width), crop=crop)

    torch.save({
        "anomaly_map": anomaly_map.cpu(),
        "distances": distances.cpu(),
        "image_paths": [str(p) for p in image_paths],
    }, "output/results.pt")


if __name__ == "__main__":
    test(n_images=None)
