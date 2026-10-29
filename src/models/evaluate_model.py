from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def main():
    data_dir = Path("data/processed_data")
    models_dir = Path("models")
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Chargement des données de test
    X_test = pd.read_csv(data_dir / "X_test_scaled.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()

    # Chargement du modèle entraîné
    model = joblib.load(models_dir / "trained_model.pkl")

    # Prédictions
    y_pred = model.predict(X_test)

    # Calcul des métriques
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    scores = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }

    # Sauvegarde des métriques au format JSON
    with open(metrics_dir / "scores.json", "w") as f:
        json.dump(scores, f, indent=4)

    # Sauvegarde des valeurs réelles et prédites
    predictions = pd.DataFrame({
        "actual": y_test,
        "predicted": y_pred
    })

    predictions.to_csv(
        data_dir / "predictions.csv",
        index=False
    )

    print("Évaluation terminée.")
    print(json.dumps(scores, indent=4))
    print("Prédictions sauvegardées dans data/processed_data/predictions.csv")


if __name__ == "__main__":
    main()
