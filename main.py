import torch
from torch.utils.data import DataLoader
from dataset import ImageFolderCustom
from settings import settings
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import yaml
import settings.YAML as yaml_settings

def main():
    with open(yaml_settings.yaml, "r") as f:
        config = yaml.safe_load(f)

    train_dir = config["data"]["train_dir"]
    test_dir = config["data"]["test_dir"]
    ground_truth_dir = config["data"]["ground_truth_dir"]
    batch_size = config["data"]["batch_size"]
    

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])