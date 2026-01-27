import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from src.model import RegressionMLP
from src.train import train_model
from src.evaluate import evaluate

def main():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results"
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(root / "output" / "X.npy")
    y = np.load(root / "output" / "y.npy")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10
    )

    scaler_X = StandardScaler().fit(X_train)
    X_train_s = scaler_X.transform(X_train)
    X_test_s  = scaler_X.transform(X_test)

    y_train_transformed = np.log1p(y_train)
    y_test_transformed  = np.log1p(y_test)

    train_ds = TensorDataset(
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(y_train_transformed, dtype=torch.float32)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test_s, dtype=torch.float32),
        torch.tensor(y_test_transformed, dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader   = DataLoader(test_ds,  batch_size=256, shuffle=False)

    model = RegressionMLP(input_dim=X.shape[1]).to(device)

    print("Training model...")
    model, train_losses, val_losses = train_model(model, train_loader, val_loader)

    target_names = ["lambda1", "mu", "psi", "lambda2", "t1"]

    print("\nEvaluating...")
    mae_test, r2_test, _, _ = evaluate(
        model, test_ds.tensors[0], y_test_transformed, target_names
    )
    mae_train, r2_train, _, _ = evaluate(
        model, train_ds.tensors[0], y_train_transformed, target_names
    )

    plt.figure(figsize=(9, 5))
    plt.plot(train_losses, label='train MSE')
    plt.plot(val_losses,   label='val MSE')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE (log scale)')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "loss_curve.png", dpi=140)
    plt.close()

    report_path = out_dir / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("MODEL PERFORMANCE REPORT\n")
        f.write("========================\n\n")

        f.write("Architecture: MLP → hidden = [256, 128], dropout=0.25, AdamW, lr=1e-4, weight_decay=1e-3\n")
        f.write(f"Device:       {device}\n")
        f.write(f"Train size:   {len(train_ds)}\n")
        f.write(f"Test size:    {len(test_ds)}\n")
        f.write("Target transform: log1p(x) on all targets\n\n")

        f.write("Metrics (test set):\n")
        f.write("--------------------\n")
        f.write(f"Average R² : {r2_test:.4f}\n")
        f.write("MAE per target:\n")
        for name, v in zip(target_names, mae_test):
            f.write(f"  {name:8} : {v:.6f}\n")

        f.write("\nMetrics (train set):\n")
        f.write("---------------------\n")
        f.write(f"Average R² : {r2_train:.4f}\n")
        f.write("MAE per target:\n")
        for name, v in zip(target_names, mae_train):
            f.write(f"  {name:8} : {v:.6f}\n")

        f.write(f"\nLoss curve saved → plots/loss_curve.png\n")

    print(f"\nReport saved to: {report_path}")
    print(f"Loss curve saved to: {plot_dir / 'loss_curve.png'}")

if __name__ == "__main__":
    main()