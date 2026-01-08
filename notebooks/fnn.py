import pandas
import numpy as np
from pathlib import Path
import numpy
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error,r2_score


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]  

    X_path = PROJECT_ROOT / "data" / "processed" / "X.npy"
    y_path = PROJECT_ROOT / "data" / "processed" / "y.npy"

    X = np.load(X_path)
    y = np.load(y_path)

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

    x_model=Pipeline([
        ("scaler",StandardScaler()),
        ("model",MLPRegressor(
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
    model=TransformedTargetRegressor(
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

    with open(results_path, "w") as f:
        f.write("=== MODEL ARCHITECTURE & PARAMETERS ===\n")
        mlp_params = model.regressor_.named_steps['model'].get_params()
        for param, value in mlp_params.items():
            f.write(f"{param}: {value}\n")
        
        f.write("\n=== PERFORMANCE METRICS ===\n")
        f.write(f"Test MAE (L1, mu, psi, L2): {test_mae}\n")
        f.write(f"Train MAE (L1, mu, psi, L2): {train_mae}\n")
        f.write(f"Test R2 Score:  {test_r2:.4f}\n")
        f.write(f"Train R2 Score: {train_r2:.4f}\n")

    print(f"Results successfully saved to {results_path}")

if __name__ == "__main__":
    main()







