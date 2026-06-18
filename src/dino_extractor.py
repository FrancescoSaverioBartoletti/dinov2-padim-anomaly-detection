import torch
import timm

PATCH_SIZE = 14


def load_model(model_name: str, device):
    model = timm.create_model(model_name, pretrained=True, num_classes=0, dynamic_img_size=True)
    model = model.to(device)
    model.eval()
    print(f"Modello caricato: {model_name}")
    return model


def _patch_spatial(tokens, n_prefix, H_patch, W_patch):
    cls_token = tokens[:, 0, :]
    patch_tokens = tokens[:, n_prefix:, :] - cls_token.unsqueeze(1)
    B, N, C = patch_tokens.shape
    return patch_tokens.reshape(B, H_patch, W_patch, C).permute(0, 3, 1, 2)


def extract_dino_features(model, dataloader, device):
    model.eval()
    all_features = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            n_prefix = model.num_prefix_tokens
            tokens = model.forward_features(images)
            H_patch = images.shape[2] // PATCH_SIZE
            W_patch = images.shape[3] // PATCH_SIZE
            all_features.append(_patch_spatial(tokens, n_prefix, H_patch, W_patch).cpu())

    return torch.cat(all_features, dim=0)
