from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def main():
    data_dir = Path("data/processed_data")
    models_dir = Path("models")

    # Chargement des données d'entraînement
    X_train = pd.read_csv(data_dir / "X_train_scaled.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()

    # Chargement des meilleurs hyperparamètres
    best_params = joblib.load(models_dir / "best_params.pkl")

    # Création du modèle avec les meilleurs paramètres
    model = RandomForestRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1
    )

    # Entraînement du modèle
    model.fit(X_train, y_train)

    # Sauvegarde du modèle entraîné
    joblib.dump(model, models_dir / "trained_model.pkl")

    print("Entraînement terminé.")
    print("Paramètres utilisés :", best_params)
    print("Modèle sauvegardé dans models/trained_model.pkl")


if __name__ == "__main__":
    main()
