import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def rechercheCoef(CSV_entrainement, model, variables=None, target='log_price', plot=False):
    df = pd.read_csv(CSV_entrainement)

    # Sélection automatique des variables numériques si non spécifiées
    if variables is None:
        variables = df.select_dtypes(include=np.number).columns.tolist()
        variables = [v for v in variables if v != target]

    # Nettoyage des données
    df_clean = df.dropna(subset=variables + [target])
    X = df_clean[variables]
    y = df_clean[target]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement
    model.fit(X_train, y_train)

    # Prédiction
    y_pred = model.predict(X_test)
    remc = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"REMC (Erreur quadratique moyenne) : {remc:.4f}")

    # Coefficients si disponibles
    coefficients = None
    if hasattr(model, 'coef_'):
        coefficients = model.coef_.tolist()
        print("Pondérations apprises :")
        for var, coef in zip(variables, coefficients):
            print(f" - {var} : {coef:.4f}")
    elif hasattr(model, 'feature_importances_'):
        coefficients = model.feature_importances_.tolist()
        print("Importance des variables (arbres) :")
        for var, coef in zip(variables, coefficients):
            print(f" - {var} : {coef:.4f}")
    else:
        print("Ce modèle ne fournit pas de coefficients/importance directe.")

    # Graphique
    if plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.xlabel("log_price réel")
        plt.ylabel("log_price prédit")
        plt.title("Prédictions vs Réel")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return model, remc, coefficients
