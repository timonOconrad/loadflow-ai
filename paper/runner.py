# sweep.py
import random
import json
import os
import matplotlib.pyplot as plt
from MLPandGNN_main import run_experiment

parquet_file = r"C:\Users\fe38cypi\Documents\git\loadflow-ai\data\updated_parquet_file.parquet"

# -----------------------
# Zufällige Config-Generierung (nur Trainings-Parameter)
# -----------------------
def sample_config():
    """return {
        "epochs": 50,
        "batch_size": random.choice([16, 32, 64, 128]),
        #"max_rows": random.choice([100, 500, 1000, 5000, 10000]),
        "max_rows": random.choice([100, 500, 1000, 5000, 10000, 50000,100000,500000]),
        "lr": random.choice([1e-1, 1e-2, 1e-3, 1e-4]),
    }"""
    return {
        "epochs": 50,
        "batch_size": random.choice([32]),
        "max_rows": random.choice([100]),
        "lr": random.choice([1e-3, 1e-4]),
    }

# -----------------------
# Lade bestehende Ergebnisse (falls vorhanden)
# -----------------------
results_file = "results_exportmodel.json"
if os.path.exists(results_file):
    with open(results_file, "r") as f:
        results = json.load(f)
else:
    results = []

# Hilfsfunktion: Prüfen, ob Config schon existiert
def config_exists(cfg, results):
    for r in results:
        if r["config"] == cfg:
            return True
    return False

# -----------------------
# Anzahl gewünschter neuer Runs
# -----------------------
N_NEW_RUNS = 10


new_runs = 0

while new_runs < N_NEW_RUNS:
    cfg = sample_config()

    # Überspringen, falls schon vorhanden
    if config_exists(cfg, results):
        print(f"Config {cfg} schon vorhanden, neue wird gezogen...")
        continue

    print(f"\n===== Neuer Run {len(results)+1} | Config: {cfg} =====")
    res = run_experiment(parquet_file, cfg)

    # -----------------------
    # Loss-Kurven speichern
    # -----------------------
    plt.figure(figsize=(10, 6))
    for model_name, loss_dict in res["loss_curves"].items():
        plt.plot(loss_dict["train"], label=f"{model_name} Train")
        plt.plot(loss_dict["val"], label=f"{model_name} Val", linestyle="--")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Training/Validation Loss | Run {len(results)+1}")
    plt.legend()
    plt.grid()

    # Hyperparam-String für Dateinamen
    cfg_str = "_".join([f"{k}{v}" for k, v in cfg.items()])
    cfg_str = cfg_str.replace(".", "p")  # z.B. 0.001 -> 0p001

    filename = f"pic/loss_curve_run{len(results)+1}_{cfg_str}.png"
    plt.savefig(filename)
    plt.close()

    # -----------------------
    # Test-Metriken ausgeben
    # -----------------------
    print("Test metrics:")
    for model, metrics in res["test_metrics"].items():
        print(f"  {model.upper()} → MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}")

    # Ergebnis speichern
    results.append({"config": cfg, "result": res})

    # Ergebnisse fortlaufend in JSON schreiben
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    new_runs += 1

print(f"\nAlle {N_NEW_RUNS} neuen Runs abgeschlossen. Ergebnisse in {results_file} gespeichert.")
