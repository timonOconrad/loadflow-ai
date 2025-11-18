from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import os

# ===========================================
# Zentral: Rollen-Definition (feature/target)
# ===========================================
ROLE_CONFIG = {
    1: {"feature": ["U_real", "U_imag"], "target": ["P", "Q", "U_abs"]},        # Slack
    2: {"feature": ["P", "U_abs"], "target": ["Q", "U_real", "U_imag"]},        # PV
    3: {"feature": ["P", "Q"], "target": ["U_real", "U_imag", "U_abs"]},        # PQ
}

BUS_TYPE = np.array([1, 2, 3, 3, 3], dtype=np.int64)


# ===========================
# Y-Bus
# ===========================
def fixed_Ybus():
    Y = np.array([
        [0.01181474-0.03523623j, -0.00945180+0.02835539j, -0.00236295+0.00708885j, 0, 0],
        [-0.00945180+0.02835539j, 0.02047889-0.06111532j, -0.00315060+0.00945180j, -0.00315060+0.00945180j, -0.00472590+0.01417769j],
        [-0.00236295+0.00708885j, -0.00315060+0.00945180j, 0.02441714-0.07304348j, -0.01890359+0.05671078j, 0],
        [0, -0.00315060+0.00945180j, -0.01890359+0.05671078j, 0.02441714-0.07304348j, -0.00236295+0.00708885j],
        [0, -0.00472590+0.01417769j, 0, -0.00236295+0.00708885j, 0.00708885-0.02111531j],
    ], dtype=np.complex64)

    return Y.real.astype(np.float32), Y.imag.astype(np.float32)


# ==========================================
# Hilfsfunktion: Dimensionen berechnen
# ==========================================
def compute_io_dims(bus_type, role_config):
    in_dim_total = sum(len(role_config[b]["feature"]) for b in bus_type)
    out_dim_total = sum(len(role_config[b]["target"]) for b in bus_type)
    per_bus_in = {b: len(role_config[b]["feature"]) for b in role_config}
    per_bus_out = {b: len(role_config[b]["target"]) for b in role_config}
    return in_dim_total, out_dim_total, per_bus_in, per_bus_out


# ================================
# Datensplits
# ================================
def make_splits_from_parquet(parquet_file, bus_type, max_rows=10000,
                             splits=(0.7, 0.15, 0.15), shuffle=True, seed=42):
    df = pd.read_parquet(parquet_file)

    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if max_rows is not None:
        df = df.iloc[:min(max_rows, len(df))]

    n = len(df)
    n_train = int(n * splits[0])
    n_val = int(n * splits[1])

    df_train, df_val, df_test = df[:n_train], df[n_train:n_train+n_val], df[n_train+n_val:]

    ds_train = UnifiedPowerDataset(df_train, bus_type, fit_scalers=True)
    ds_val   = UnifiedPowerDataset(df_val, bus_type, scalers=ds_train.scalers)
    ds_test  = UnifiedPowerDataset(df_test, bus_type, scalers=ds_train.scalers)

    return ds_train, ds_val, ds_test

# ======================
# Dataset
# ======================
class UnifiedPowerDataset(Dataset):
    def __init__(self, df, bus_type, role_config=ROLE_CONFIG, fit_scalers=False, scalers=None):
        self.df = df.reset_index(drop=True)
        self.bus_type = np.array(bus_type, dtype=np.int64)
        self.role_config = role_config

        raw = self._preprocess_df(self.df)

        if fit_scalers:
            self.scalers = {feat: StandardScaler().fit(mat.reshape(-1, 1)) for feat, mat in raw.items()}
        else:
            self.scalers = scalers

        self.data = {
            feat: self.scalers[feat].transform(mat.reshape(-1, 1)).reshape(mat.shape)
            for feat, mat in raw.items()
        }

        Yr, Yi = fixed_Ybus()
        self.Yr = torch.from_numpy(Yr)
        self.Yi = torch.from_numpy(Yi)
        self.bus_type_tensor = torch.from_numpy(self.bus_type)

    def _preprocess_df(self, df):
        P = np.stack([np.array(r["P_G"]) + np.array(r["P_L"]) for _, r in df.iterrows()])
        Q = np.stack([np.array(r["Q_G"]) + np.array(r["Q_L"]) for _, r in df.iterrows()])
        U_real = np.stack([np.array(r["u_powerfactory_real"]) for _, r in df.iterrows()])
        U_imag = np.stack([np.array(r["u_powerfactory_imag"]) for _, r in df.iterrows()])
        U_abs  = np.sqrt(U_real**2 + U_imag**2)

        return {"P": P, "Q": Q, "U_real": U_real, "U_imag": U_imag, "U_abs": U_abs}

    def _build_bus_mats(self, idx):
        feat_rows, targ_rows = [], []

        for bus_idx, btype in enumerate(self.bus_type):
            f = [self.data[feat][idx, bus_idx] for feat in self.role_config[btype]["feature"]]
            t = [self.data[feat][idx, bus_idx] for feat in self.role_config[btype]["target"]]

            feat_rows.append(f)
            targ_rows.append(t)

        return np.array(feat_rows, dtype=np.float32), np.array(targ_rows, dtype=np.float32)

    def __getitem__(self, idx):
        feat_mat, targ_mat = self._build_bus_mats(idx)

        return {
            "gnn": {
                "bus_type": self.bus_type_tensor,
                "Ybus_real": self.Yr,
                "Ybus_imag": self.Yi,
                "features": torch.from_numpy(feat_mat),
                "targets_bus": torch.from_numpy(targ_mat),
            },
            "mlp": {
                "X": torch.from_numpy(feat_mat.flatten()),
                "Y": torch.from_numpy(targ_mat.flatten())
            },
            "target": torch.from_numpy(targ_mat.flatten()),
        }

    def __len__(self):
        return len(self.df)

    def inverse_transform_Y(self, Y_scaled):
        if isinstance(Y_scaled, torch.Tensor):
            Y_scaled = Y_scaled.detach().cpu().numpy()

        Y_scaled = Y_scaled.reshape(-1)
        targ_rows, cursor = [], 0

        for bus_idx, btype in enumerate(self.bus_type):
            row = []
            for feat in self.role_config[btype]["target"]:
                val_s = Y_scaled[cursor]
                val_o = self.scalers[feat].inverse_transform([[val_s]])[0, 0]
                row.append(val_o)
                cursor += 1
            targ_rows.append(row)

        return np.array(targ_rows, dtype=np.float32).flatten()

# =================
# Modelle
# =================
class Net(nn.Module):
    """Einfaches MLP"""
    def __init__(self, in_dim, out_dim, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ----- GNNs -----
class BusEmbedding(nn.Module):
    def __init__(self, d, role_config):
        super().__init__()
        self.d = d
        self.mlps = nn.ModuleDict()
        for b, conf in role_config.items():
            self.mlps[str(b)] = nn.Sequential(
                nn.Linear(len(conf["feature"]), d),
                nn.Tanh()
            )

    def forward(self, feat, bus_type, bus_feature_dims):
        out = torch.zeros(feat.size(0), self.d, device=feat.device)
        for b_str, mlp in self.mlps.items():
            b = int(b_str)
            mask = bus_type == b
            if mask.any():
                out[mask] = mlp(feat[mask, :bus_feature_dims[b]])
        return out


class Leap(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, H, A):
        return H + torch.tanh(self.lin(torch.matmul(A, H)))


class Decoder(nn.Module):
    def __init__(self, d, out_dim):
        super().__init__()
        self.head = nn.Linear(d, out_dim)

    def forward(self, H):
        return self.head(H.mean(dim=1))


class BusDecoder(nn.Module):
    def __init__(self, d, role_config):
        super().__init__()
        self.decoders = nn.ModuleDict({
            str(b): nn.Linear(d, len(c["target"])) for b, c in role_config.items()
        })

    def forward(self, H, bus_type):
        return torch.cat(
            [self.decoders[str(int(bus_type[0, n].item()))](H[:, n, :]) for n in range(H.size(1))],
            dim=-1
        )


class GNSSolver(nn.Module):
    """GNN mit globalem Decoder"""
    def __init__(self, d=100, K=10, out_dim=15, role_config=ROLE_CONFIG, bus_feature_dims=None):
        super().__init__()
        self.K = K
        self.emb = BusEmbedding(d, role_config)
        self.leap = Leap(d)
        self.dec = Decoder(d, out_dim)
        self.bus_feature_dims = bus_feature_dims

    def forward(self, bus_type, Yr, Yi, features):
        B, N, F = features.shape
        H = self.emb(
            features.reshape(B * N, F), bus_type.view(-1), self.bus_feature_dims
        ).view(B, N, -1)

        A = ((Yr != 0) | (Yi != 0)).float()
        idx = torch.arange(N, device=A.device)
        A[:, idx, idx] = 0
        A = A / A.sum(-1, keepdim=True).clamp_min(1)

        for _ in range(self.K):
            H = self.leap(H, A)

        return self.dec(H)


class GNSSolver2(nn.Module):
    """GNN mit Bus-weise Decoder"""
    def __init__(self, d=100, K=10, role_config=ROLE_CONFIG, bus_feature_dims=None):
        super().__init__()
        self.K = K
        self.emb = BusEmbedding(d, role_config)
        self.leap = Leap(d)
        self.dec = BusDecoder(d, role_config)
        self.bus_feature_dims = bus_feature_dims

    def forward(self, bus_type, Yr, Yi, features):
        B, N, F = features.shape
        H = self.emb(
            features.reshape(B * N, F), bus_type.view(-1), self.bus_feature_dims
        ).view(B, N, -1)

        A = ((Yr != 0) | (Yi != 0)).float()
        idx = torch.arange(N, device=A.device)
        A[:, idx, idx] = 0
        A = A / A.sum(-1, keepdim=True).clamp_min(1)

        for _ in range(self.K):
            H = self.leap(H, A)

        return self.dec(H, bus_type)


# ====================
# Hilfsfunktionen
# ====================
def compute_metrics(pred, true):
    """Berechnet MSE & MAE"""
    mse = torch.mean((pred - true) ** 2).item()
    mae = torch.mean(torch.abs(pred - true)).item()
    return {"mse": mse, "mae": mae}


# ====================
# Training
# ====================
def run_experiment(parquet_file, config):
    """
    Führt ein Experiment mit gegebener Konfiguration durch.
    Gibt Loss-Kurven und finale Metriken zurück.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Daten vorbereiten ---
    train_ds, val_ds, test_ds = make_splits_from_parquet(
        parquet_file,
        BUS_TYPE,
        max_rows=config.get("max_rows", 10000),
        seed=config.get("dataset_seed", 42)
    )
    in_dim, out_dim, per_bus_in, _ = compute_io_dims(BUS_TYPE, ROLE_CONFIG)

    loaders = {
        "train": DataLoader(train_ds, batch_size=config.get("batch_size", 32), shuffle=True),
        "val":   DataLoader(val_ds, batch_size=config.get("batch_size", 32)),
        "test":  DataLoader(test_ds, batch_size=config.get("batch_size", 32)),
    }

    # --- Modelle ---
    models = {
        "mlp": Net(in_dim, out_dim, hidden=config.get("hidden", 64)).to(device),
        "gnn1": GNSSolver(100, config.get("gnn_depth", 5), out_dim, ROLE_CONFIG, per_bus_in).to(device),
        "gnn2": GNSSolver2(100, config.get("gnn_depth", 5), ROLE_CONFIG, per_bus_in).to(device),
    }

    # --- Optimizer ---
    opts = {k: optim.Adam(m.parameters(), lr=config.get("lr", 1e-3)) for k, m in models.items()}
    crit = nn.MSELoss()
    losses = {k: {"train": [], "val": []} for k in models}

    epochs = config.get("epochs", 50)

    # --- Training Loop ---
    for epoch in trange(epochs, desc="Training", ncols=80):

        for m in models.values():
            m.train()
        sums = {k: 0.0 for k in models}
        for batch in loaders["train"]:
            X, Y = batch["mlp"]["X"].to(device), batch["mlp"]["Y"].to(device)
            g = batch["gnn"]
            bt, Yr, Yi, F = (
                g["bus_type"].to(device),
                g["Ybus_real"].to(device),
                g["Ybus_imag"].to(device),
                g["features"].to(device),
            )

            for k, m in models.items():
                pred = m(X) if k == "mlp" else m(bt, Yr, Yi, F)
                loss = crit(pred, Y)

                opts[k].zero_grad()
                loss.backward()
                opts[k].step()
                sums[k] += loss.item()

        # --- Validation ---
        for m in models.values():
            m.eval()
        sums_val = {k: 0.0 for k in models}
        with torch.no_grad():
            for batch in loaders["val"]:
                X, Y = batch["mlp"]["X"].to(device), batch["mlp"]["Y"].to(device)
                g = batch["gnn"]
                bt, Yr, Yi, F = (
                    g["bus_type"].to(device),
                    g["Ybus_real"].to(device),
                    g["Ybus_imag"].to(device),
                    g["features"].to(device),
                )

                for k, m in models.items():
                    pred = m(X) if k == "mlp" else m(bt, Yr, Yi, F)
                    sums_val[k] += crit(pred, Y).item()

        for k in models:
            losses[k]["train"].append(sums[k] / len(loaders["train"]))
            losses[k]["val"].append(sums_val[k] / len(loaders["val"]))
        pass
    print()
    # --- Speichern ---
    save_dir= ""
    batch_size = config.get("batch_size", 32)
    max_rows = config.get("max_rows", 10000)
    lr = config.get("lr", 1e-3)
    # --- Modelle speichern (letzte Epoche) ---
    for k, m in models.items():
        path = os.path.join(save_dir, f"{k}_bs_{batch_size}_rows_{max_rows}_lr_{lr}.pt")
        torch.save(m.state_dict(), path)
        print(f"[INFO] Modell gespeichert unter {path}")

    # --- Test ---
    test_metrics = {k: {"mse": 0.0, "mae": 0.0} for k in models}
    with torch.no_grad():
        for k, m in models.items():
            m.eval()
            mse_sum, mae_sum, n = 0.0, 0.0, 0
            for batch in loaders["test"]:
                X, Y = batch["mlp"]["X"].to(device), batch["mlp"]["Y"].to(device)
                g = batch["gnn"]
                bt, Yr, Yi, F = (
                    g["bus_type"].to(device),
                    g["Ybus_real"].to(device),
                    g["Ybus_imag"].to(device),
                    g["features"].to(device),
                )
                pred = m(X) if k == "mlp" else m(bt, Yr, Yi, F)
                metrics = compute_metrics(pred, Y)
                mse_sum += metrics["mse"]
                mae_sum += metrics["mae"]
                n += 1
            test_metrics[k]["mse"] = mse_sum / n
            test_metrics[k]["mae"] = mae_sum / n

    return {
        "loss_curves": losses,
        "final_val": {k: losses[k]["val"][-1] for k in models},
        "test_metrics": test_metrics,
    }
