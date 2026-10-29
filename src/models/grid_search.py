from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def main():
    data_dir = Path("data/processed_data")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Chargement des données d'entraînement
    X_train = pd.read_csv(data_dir / "X_train_scaled.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()

    # Modèle de régression
    model = RandomForestRegressor(random_state=42, n_jobs=1)

    # Hyperparamètres à tester
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5]
    }

    # Recherche des meilleurs hyperparamètres
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # Sauvegarde des meilleurs paramètres
    joblib.dump(
        grid_search.best_params_,
        models_dir / "best_params.pkl"
    )

    print("GridSearch terminé.")
    print("Meilleurs paramètres :", grid_search.best_params_)
    print("Meilleur MSE (validation croisée) :", -grid_search.best_score_)


if __name__ == "__main__":
    main()
