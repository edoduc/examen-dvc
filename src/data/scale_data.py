from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


def main():
    data_dir = Path("data/processed_data")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Chargement des données d'entraînement et de test
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")

    # Ajustement uniquement sur l'ensemble d'entraînement
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Conservation des noms de colonnes
    X_train_scaled = pd.DataFrame(
        X_train_scaled, columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled, columns=X_test.columns
    )

    # Sauvegarde des données normalisées
    X_train_scaled.to_csv(
        data_dir / "X_train_scaled.csv", index=False
    )
    X_test_scaled.to_csv(
        data_dir / "X_test_scaled.csv", index=False
    )

    # Sauvegarde du scaler pour de futures prédictions
    joblib.dump(scaler, models_dir / "scaler.pkl")

    print("Normalisation terminée.")
    print(f"X_train_scaled : {X_train_scaled.shape}")
    print(f"X_test_scaled  : {X_test_scaled.shape}")


if __name__ == "__main__":
    main()
