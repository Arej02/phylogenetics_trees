import pandas as pd
import numpy as np
from matplotlib.pylab import plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    X_path = PROJECT_ROOT / "data" / "simulated" / "X.npy"
    y_path = PROJECT_ROOT / "data" / "simulated" / "y.npy"

    X = np.load(X_path)
    y = np.load(y_path)

    assert y.shape[1] == 5, f"Expected y to have 5 columns, but got {y.shape[1]}"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10
    )

    x_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(512, 256),
            activation="relu",
            solver="adam",
            max_iter=200,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=10,
            verbose=True
        ))
    ])

    model = TransformedTargetRegressor(
        regressor=x_model,
        transformer=StandardScaler()
    )

    model.fit(X_train, y_train)

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "model_report.txt"

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    test_mae = mean_absolute_error(y_test, y_pred_test, multioutput="raw_values")
    train_mae = mean_absolute_error(y_train, y_pred_train, multioutput="raw_values")

    test_r2 = r2_score(y_test, y_pred_test, multioutput="uniform_average")
    train_r2 = r2_score(y_train, y_pred_train, multioutput="uniform_average")

    target_names = ["lambda1", "mu", "psi", "lambda2", "t1"]

    with open(results_path, "w") as f:
        f.write("=== MODEL ARCHITECTURE & PARAMETERS ===\n")
        mlp_params = model.regressor_.named_steps["model"].get_params()
        for param, value in mlp_params.items():
            f.write(f"{param}: {value}\n")

        f.write("\n=== PERFORMANCE METRICS ===\n")

        f.write("Test MAE per target:\n")
        for name, val in zip(target_names, test_mae):
            f.write(f"  {name}: {val:.6f}\n")

        f.write("\nTrain MAE per target:\n")
        for name, val in zip(target_names, train_mae):
            f.write(f"  {name}: {val:.6f}\n")

        f.write(f"\nTest R2 Score (avg):  {test_r2:.4f}\n")
        f.write(f"Train R2 Score (avg): {train_r2:.4f}\n")

    print(f"Results successfully saved to {results_path}")

    plot_dir = PROJECT_ROOT / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    mlp = model.regressor_.named_steps["model"]

    plt.figure(figsize=(10, 5))
    plt.plot(mlp.loss_curve_)
    plt.title("Model Loss Curve (Training)")
    plt.xlabel("Iterations")
    plt.ylabel("Loss (MSE)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(plot_dir / "loss_curve.png")
    plt.close()

    if hasattr(mlp, "validation_scores_"):
        plt.figure(figsize=(10, 5))
        plt.plot(mlp.validation_scores_)
        plt.title("Validation Score (R²) during Training")
        plt.xlabel("Iterations")
        plt.ylabel("R² Score")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(plot_dir / "validation_score_curve.png")
        plt.close()

    print(f"Plots saved in: {plot_dir}")


if __name__ == "__main__":
    main()
