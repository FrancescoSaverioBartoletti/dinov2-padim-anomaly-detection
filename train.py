import yaml
import torchvision.transforms as transforms
from src.dataset import ToothbrushImageDataset, CropSides
from torch.utils.data import DataLoader, Subset
import torch
from src.dino_extractor import load_model, extract_dino_features
from src.padim import train_padim, calculate_mahalanobis_distance, save_padim_stats

def train(debug: bool = False):
    with open("settings.YAML", "r") as f:
        config = yaml.safe_load(f)

    train_dir = config["data"]["train_dir"]
    test_dir = config["data"]["test_dir"]
    ground_truth_dir = config["data"]["ground_truth_dir"]
    batch_size = config["data"]["batch_size"]
    height = config["data"]["height"]
    width = config["data"]["width"]

    crop = config["data"].get("crop", {})
    crop_transform = [CropSides(**crop)] if crop else []
    train_transform = transforms.Compose(crop_transform + [
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = ToothbrushImageDataset(train_dir, train_transform)

    if debug:
        n = min(50, len(train_dataset))
        train_dataset = Subset(train_dataset, range(n))
        print(f"[DEBUG] Subdataset: {n} immagini")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    backbone = config["model"]["backbone"]
    d_reduced = config["model"]["d_reduced"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(backbone, device)

    features = extract_dino_features(model, train_dataloader, device)

    C = features.shape[1]
    torch.manual_seed(42)
    selected_channels = torch.randperm(C)[:d_reduced]
    features_reduced = features[:, selected_channels, :, :]
    print(f"Feature ridotte: {C} -> {d_reduced} canali")

    mean, inv_cov = train_padim(features_reduced)

    train_distances = calculate_mahalanobis_distance(features_reduced, mean, inv_cov)

    tau_matrix = torch.quantile(train_distances, q=0.999, dim=0)
    save_padim_stats(mean, inv_cov, tau_matrix, selected_channels)
    print(f"Train completato. tau_matrix shape: {tau_matrix.shape}")

if __name__ == "__main__":
    train(debug=False)

