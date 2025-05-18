import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def rechercheCoef(CSV_entrainement, variables=None, target='log_price', plot=False):
    
    df = pd.read_csv(CSV_entrainement)

    # Sélection automatique des variables numériques si non fournies
    if variables is None:
        variables = df.select_dtypes(include=np.number).columns.tolist()
        variables = [v for v in variables if v != target]

    df_clean = df.dropna(subset=variables + [target])
    X = df_clean[variables]
    y = df_clean[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédiction
    y_pred = model.predict(X_test)
    remc = np.sqrt(mean_squared_error(y_test, y_pred))

    # Coefficients dans le même ordre que les variables
    coefficients = model.coef_.tolist()  # liste ordonnée
    print(f"\nREMC (Erreur quadratique moyenne) : {remc:.4f}")
    print("\nPondérations apprises :")
    for var, coef in zip(variables, coefficients):
        print(f" - {var} : {coef:.4f}")

    # Graphique optionnel
    if plot:
        import matplotlib.pyplot as plt
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.xlabel("log_price réel")
        plt.ylabel("log_price prédit")
        plt.title("Prédictions vs Réel")
        plt.grid(True)
        plt.show()

    return model, remc, coefficients
