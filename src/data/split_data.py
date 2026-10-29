from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    # Définition des chemins
    input_path = Path("data/raw_data/raw.csv")
    output_dir = Path("data/processed_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Chargement des données
    df = pd.read_csv(input_path)

    # Séparation des variables explicatives et de la cible
    X = df.drop(columns=["date", "silica_concentrate"])
    y = df[["silica_concentrate"]]

    # Split aléatoire : 80 % train, 20 % test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Sauvegarde des quatre datasets
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    print("Séparation des données terminée.")
    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")
    print(f"y_train : {y_train.shape}")
    print(f"y_test  : {y_test.shape}")


if __name__ == "__main__":
    main()
