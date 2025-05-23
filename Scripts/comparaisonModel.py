from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import pandas as pd

def comparer_models(csv_path, target='log_price', variables=None, plot=False, show_coef=False):
    from recherche_coef import rechercheCoef  # ou directement intégrer la fonction dans ce script

    modèles = {
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(kernel='rbf'),
        "MLPRegressor": MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
    }

    résultats = []

    for nom, modèle in modèles.items():
        print(f"Test du modèle : {nom}")
        model_fitted, remc, coef = rechercheCoef(
            CSV_entrainement=csv_path,
            model=modèle,
            variables=variables,
            target=target,
            plot=plot if nom == "XGBoost" else False  # afficher qu'un seul plot par défaut
        )
        if show_coef:
            if coef:
                print(f" → {len(coef)} coefficients / importances affichés.")
            else:
                print(" → Pas de coefficients disponibles.")
        résultats.append((nom, remc))

    # Résumé
    print("Résumé des performances (trié par REMC croissant) :")
    résultats.sort(key=lambda x: x[1])
    for nom, rmse in résultats:
        print(f" - {nom:<15} : RMSE = {rmse:.4f}")

    return résultats
