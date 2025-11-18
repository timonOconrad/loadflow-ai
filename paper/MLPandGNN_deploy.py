import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import time

from MLPandGNN_main import (
    Net, GNSSolver, GNSSolver2,
    ROLE_CONFIG, BUS_TYPE,
    compute_io_dims,
    make_splits_from_parquet,
    compute_metrics
)

# ==========================
# Modelle laden
# ==========================
def load_all_models(save_dir="saved_models", hidden=64, gnn_depth=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim, out_dim, per_bus_in, _ = compute_io_dims(BUS_TYPE, ROLE_CONFIG)

    # Instanzen erzeugen
    models = {
        "mlp": Net(in_dim, out_dim, hidden=hidden).to(device),
        "gnn1": GNSSolver(100, gnn_depth, out_dim, ROLE_CONFIG, per_bus_in).to(device),
        "gnn2": GNSSolver2(100, gnn_depth, ROLE_CONFIG, per_bus_in).to(device),
    }

    # Alle Dateien im Ordner durchgehen
    for fname in os.listdir(save_dir):
        fpath = os.path.join(save_dir, fname)
        if "mlp" in fname:
            models["mlp"].load_state_dict(torch.load(fpath, map_location=device))
            print(f"[INFO] Modell mlp erfolgreich geladen aus {fpath}")
        elif "gnn1" in fname:
            models["gnn1"].load_state_dict(torch.load(fpath, map_location=device))
            print(f"[INFO] Modell gnn1 erfolgreich geladen aus {fpath}")
        elif "gnn2" in fname:
            models["gnn2"].load_state_dict(torch.load(fpath, map_location=device))
            print(f"[INFO] Modell gnn2 erfolgreich geladen aus {fpath}")

    # Alle auf eval
    for m in models.values():
        m.eval()

    return models


# ==========================
# Inferenz
# ==========================
def run_inference(parquet_file, save_dir="saved_models", batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- gleiche Splits wie im Training ---
    _, _, test_ds = make_splits_from_parquet(parquet_file, BUS_TYPE, seed=42)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    models = load_all_models(save_dir)

    # Rückskalierte Werte sammeln
    all_true = []
    all_preds = {k: [] for k in models}

    with torch.no_grad():
        for batch in test_loader:
            X, Y = batch["mlp"]["X"].to(device), batch["mlp"]["Y"].to(device)
            g = batch["gnn"]
            bt, Yr, Yi, F = (
                g["bus_type"].to(device),
                g["Ybus_real"].to(device),
                g["Ybus_imag"].to(device),
                g["features"].to(device),
            )

            # Groundtruth zurückskalieren
            Y_inv = [test_ds.inverse_transform_Y(y) for y in Y.cpu()]
            all_true.append(np.stack(Y_inv))

            for k, m in models.items():
                if "mlp" in k:
                    pred = m(X)
                else:
                    pred = m(bt, Yr, Yi, F)

                pred_inv = [test_ds.inverse_transform_Y(p) for p in pred.cpu()]
                all_preds[k].append(np.stack(pred_inv))

    all_true = np.concatenate(all_true, axis=0)
    for k in all_preds:
        all_preds[k] = np.concatenate(all_preds[k], axis=0)

    # --- Feature-Reihenfolge bestimmen ---
    feature_names = []
    for b in BUS_TYPE:
        feature_names.extend(ROLE_CONFIG[b]["target"])

    feature_names = [f"Bus{i}_{name}" for i, b in enumerate(BUS_TYPE) for name in ROLE_CONFIG[b]["target"]]

    # --- Metriken pro Feature ---
    results = {k: {} for k in models}
    for k in models:
        for i, fname in enumerate(feature_names):
            y_true = all_true[:, i]
            y_pred = all_preds[k][:, i]
            mse = np.mean((y_pred - y_true) ** 2)
            mae = np.mean(np.abs(y_pred - y_true))
            results[k][fname] = {"mse": mse, "mae": mae}
        # --- Zufälliges Batch zur Ansicht ---
        rand_idx = random.randint(0, len(test_ds) - 1)
        sample = test_ds[rand_idx]

        X = sample["mlp"]["X"].unsqueeze(0).to(device)
        Y = sample["mlp"]["Y"].unsqueeze(0).to(device)
        g = sample["gnn"]
        bt, Yr, Yi, F = (
            g["bus_type"].unsqueeze(0).to(device),
            g["Ybus_real"].unsqueeze(0).to(device),
            g["Ybus_imag"].unsqueeze(0).to(device),
            g["features"].unsqueeze(0).to(device),
        )

        print("\n===== Zufalls-Sample (rückskaliert) =====")
        y_true = test_ds.inverse_transform_Y(Y[0])
        print("Groundtruth:", y_true)

        for k, m in models.items():
            if "mlp" in k:
                pred = m(X)
            else:
                pred = m(bt, Yr, Yi, F)
            y_pred = test_ds.inverse_transform_Y(pred[0])
            print(f"{k} Prediction:", y_pred)

    times = measure_inference_time(models, test_ds, device)
    print("\n===== Inferenzzeiten =====")
    for k in times:
        print(f"\nModell {k}:")
        for n, t in times[k].items():
            print(f"  {n:5d} Samples: {t * 1000:.12f} ms")

    print("\n===== Metriken pro Feature =====")
    for k in models:
        print(f"\nModell {k}:")
        for fname in feature_names:
            mse = results[k][fname]["mse"]
            mae = results[k][fname]["mae"]
            print(f"  {fname:12s} MSE={mse:.4f}, MAE={mae:.4f}")

    # --- Plots ---
    for k in models:
        # Fehlerhistogramme pro Feature
        n_feat = len(feature_names)
        n_cols = 3
        n_rows = int(np.ceil(n_feat / n_cols))
        plt.figure(figsize=(5*n_cols, 3*n_rows))
        for i, fname in enumerate(feature_names):
            y_true = all_true[:, i]
            y_pred = all_preds[k][:, i]
            errors = y_pred - y_true
            plt.subplot(n_rows, n_cols, i+1)
            plt.hist(errors, bins=50, alpha=0.7)
            plt.title(f"{k}: {fname}")
            plt.xlabel("y_pred - y_true")
            plt.ylabel("Häufigkeit")
        plt.tight_layout()
        plt.savefig(f"{k}_delta.png")
        plt.show()

        # Scatterplots pro Feature
        plt.figure(figsize=(5*n_cols, 3*n_rows))
        for i, fname in enumerate(feature_names):
            y_true = all_true[:500, i]
            y_pred = all_preds[k][:500, i]
            plt.subplot(n_rows, n_cols, i+1)
            plt.scatter(y_true, y_pred, alpha=0.5)
            lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
            plt.plot(lims, lims, "r--")
            plt.title(f"{k}: {fname}")
            plt.xlabel("y_true")
            plt.ylabel("y_pred")

        plt.tight_layout()
        plt.savefig(f"{k}_abs.png")
        plt.show()




    return results, all_true, all_preds




def measure_inference_time(
        models,
        test_ds,
        device,
        n_samples_list=[1,5,10,50, 100, 500, 1000, 5000, 10000],
        repeats=20,
):
    """
    Inferenzzeit (Forward-Pass)
    """
    results = {k: {} for k in models}

    for n in n_samples_list:
        # Wenn n größer als Dataset → replace=True
        idxs = np.random.choice(len(test_ds), size=n, replace=(n > len(test_ds)))
        batch = [test_ds[i] for i in idxs]

        # Manuelles Stapeln
        X = torch.stack([b["mlp"]["X"] for b in batch]).to(device)
        Y = torch.stack([b["mlp"]["Y"] for b in batch]).to(device)
        bt = torch.stack([b["gnn"]["bus_type"] for b in batch]).to(device)
        Yr = torch.stack([b["gnn"]["Ybus_real"] for b in batch]).to(device)
        Yi = torch.stack([b["gnn"]["Ybus_imag"] for b in batch]).to(device)
        F = torch.stack([b["gnn"]["features"] for b in batch]).to(device)

        # Warmup
        with torch.no_grad():
            for k, m in models.items():
                if "mlp" in k:
                    _ = m(X)
                else:
                    _ = m(bt, Yr, Yi, F)

        # Zeitmessung mit Wiederholungen
        with torch.no_grad():
            for k, m in models.items():
                total = 0.0
                for _ in range(repeats):
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    if "mlp" in k:
                        _ = m(X)
                    else:
                        _ = m(bt, Yr, Yi, F)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    total += (t1 - t0)
                avg_ms = (total / repeats) * 1000.0  # in ms
                results[k][n] = avg_ms

    return results


if __name__ == "__main__":
    parquet_file = r"C:\Users\fe38cypi\Documents\git\loadflow-ai\data\updated_parquet_file.parquet"
    run_inference(parquet_file)

"""
===== Inferenzzeiten =====

MModell mlp:
      1 Samples: 47.120000000689 ms
      5 Samples: 49.820000000977 ms
     10 Samples: 48.289999999795 ms
     50 Samples: 53.959999999620 ms
    100 Samples: 69.710000000001 ms
    500 Samples: 141.799999999392 ms
   1000 Samples: 142.770000000425 ms
   5000 Samples: 207.319999999811 ms
  10000 Samples: 350.610000000273 ms

Modell gnn1:
      1 Samples: 900.430000000085 ms
      5 Samples: 979.789999999880 ms
     10 Samples: 1109.419999999872 ms
     50 Samples: 1402.880000000195 ms
    100 Samples: 1736.950000000093 ms
    500 Samples: 3042.389999999528 ms
   1000 Samples: 3648.179999999357 ms
   5000 Samples: 16242.400000000502 ms
  10000 Samples: 27734.839999999393 ms

Modell gnn2:
      1 Samples: 1082.200000000455 ms
      5 Samples: 1213.370000000680 ms
     10 Samples: 1360.849999999303 ms
     50 Samples: 1696.080000000322 ms
    100 Samples: 1984.910000000184 ms
    500 Samples: 3875.129999999416 ms
   1000 Samples: 3791.410000000539 ms
   5000 Samples: 16386.749999999851 ms
  10000 Samples: 30706.250000000779 ms
  
===== Newton-Raphson Benchmark (Gesamtzeit) =====
     1 Samples: 22.004 ms
     5 Samples: 69.004 ms
    10 Samples: 157.990 ms
    50 Samples: 693.989 ms
   100 Samples: 1494.972 ms
   500 Samples: 7173.792 ms
  1000 Samples: 14049.903 ms
  5000 Samples: 70335.630 ms
 10000 Samples: 142490.457 ms
"""



