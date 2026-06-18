import torch
from pathlib import Path

def train_padim(features, epsilon=0.01):
    """
    Calcola la media e l'inversa della matrice di covarianza per ogni patch spaziale.

    Args:
        features (torch.Tensor): Tensore in input di forma (Batch, Canali, Altezza, Larghezza).
        epsilon (float): Termine di regolarizzazione per garantire l'invertibilità.

    Returns:
        mean (torch.Tensor): Medie per ogni patch, forma (Canali, Altezza * Larghezza).
        inv_cov (torch.Tensor): Matrici di covarianza inverse per ogni patch, 
                                forma (Canali, Canali, Altezza * Larghezza).
    """
    B, C, H, W = features.shape
    N = H * W
    
    features_flat = features.view(B, C, N)
    mean = torch.mean(features_flat, dim=0)
    
    # Inizializza il tensore che conterrà le inverse delle matrici di covarianza
    inv_cov = torch.zeros(C, C, N, device=features.device)
    identity = torch.eye(C, device=features.device)
    
    for i in range(N):
        patch_features = features_flat[:, :, i]
        patch_mean = mean[:, i].unsqueeze(0)
        centered_features = patch_features - patch_mean
        
        # Calcolo della covarianza
        patch_cov = torch.matmul(centered_features.t(), centered_features) / (B - 1)
        patch_cov += epsilon * identity
        
        # Calcolo dell'inversa direttamente in questa fase
        patch_inv_cov = torch.linalg.inv(patch_cov)
        
        # Salva l'inversa nel tensore finale
        inv_cov[:, :, i] = patch_inv_cov
        
    return mean, inv_cov

def calculate_mahalanobis_distance(test_features, train_mean, train_inv_cov):
    """
    Calcola la distanza di Mahalanobis utilizzando l'inversa della covarianza pre-calcolata.
    Vettorizzato su tutti i patch contemporaneamente tramite einsum.
    """
    B, C, H, W = test_features.shape
    N = H * W

    delta = test_features.view(B, C, N) - train_mean.unsqueeze(0)  # (B, C, N)
    # left_term[b, d, n] = sum_c delta[b,c,n] * inv_cov[c,d,n]
    left_term = torch.einsum('bcn,cdn->bdn', delta, train_inv_cov)  # (B, C, N)
    sq_dist = (left_term * delta).sum(dim=1).clamp(min=0)           # (B, N)
    return sq_dist.sqrt().view(B, H, W)

def save_padim_stats(mean, inv_cov, tau_matrix, selected_channels, model_dir: str = "model"):
    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "padim_stats.pt"
    tmp_path = out_path.with_suffix(".tmp")
    torch.save({"mean": mean, "inv_cov": inv_cov, "tau_matrix": tau_matrix, "selected_channels": selected_channels}, tmp_path)
    tmp_path.replace(out_path)  # atomico: visibile solo quando completo
    print(f"Stats salvate in {out_path}")

def sigmoid_normalize(anomaly_map, tau_matrix, k: float = 0.08):
    # s = 1 / (1 + k * exp(tau - distance))
    return 1.0 / (1.0 + k * torch.exp(tau_matrix - anomaly_map))
