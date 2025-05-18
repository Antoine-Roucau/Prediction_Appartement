import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# 1. Données
xi = np.arange(1, 31)
yi = np.array([17.072616, 17.233056, 15.819026, 9.059424, 8.366792, 15.539386, 18.953472, 18.950901, 22.056480, 20.058018,
               15.548149, 19.143198, 24.392974, 27.732440, 31.631984, 22.653058, 22.899376, 24.965741, 30.366796, 32.851772,
               33.834252, 33.774935, 27.962827, 26.106150, 36.793030, 39.424127, 42.248909, 39.446712, 37.097101, 33.232282])

# Créer le DataFrame
df = pd.DataFrame({'x': xi, 'y': yi})
print("Données importées:")
print(df.head(10))
print(f"Nombre d'observations: {len(df)}")

# 2. Fonction pour créer la matrice de design pour chaque modèle
def create_design_matrix(x, model_type):
    n = len(x)
    
    if model_type == 1:  # y = β₀ + β₁x
        X = np.column_stack([np.ones(n), x])
    elif model_type == 2:  # y = β₀ + β₁x + β₂√x
        X = np.column_stack([np.ones(n), x, np.sqrt(x)])
    elif model_type == 3:  # y = β₀ + β₁x + β₂x²
        X = np.column_stack([np.ones(n), x, x**2])
    elif model_type == 4:  # y = β₀ + β₁x + β₂ln(x)
        X = np.column_stack([np.ones(n), x, np.log(x)])
    elif model_type == 5:  # y = β₀ + β₁x + β₂√(x+10)
        X = np.column_stack([np.ones(n), x, np.sqrt(x + 10)])
    elif model_type == 6:  # y = β₀ + β₁x + β₂ln(x) + β₃x³
        X = np.column_stack([np.ones(n), x, np.log(x), x**3])
    elif model_type == 7:  # y = β₀ + β₁x + β₂cos(x)
        X = np.column_stack([np.ones(n), x, np.cos(x)])
    elif model_type == 8:  # y = β₀ + β₁x² + β₂e^x
        X = np.column_stack([np.ones(n), x**2, np.exp(x)])
    elif model_type == 9:  # y = β₀ + β₁x + β₂sin(x)
        X = np.column_stack([np.ones(n), x, np.sin(x)])
    elif model_type == 10:  # y = β₀ + β₁x + β₂sin(x) + β₃x²
        X = np.column_stack([np.ones(n), x, np.sin(x), x**2])
    elif model_type == 11:  # y = β₀ + β₁x² + β₂tan(x) + β₃e^x
        X = np.column_stack([np.ones(n), x**2, np.tan(x), np.exp(x)])
    elif model_type == 12:  # y = β₀ + β₁x² + β₂cos(x) + β₃sin(x)
        X = np.column_stack([np.ones(n), x**2, np.cos(x), np.sin(x)])
    
    return X

# 3. Fonction pour calculer le R² ajusté
def adjusted_r2(r2, n, p):
    return 1 - ((n - 1) / (n - p - 1)) * (1 - r2)

# 4. Analyser tous les modèles
model_descriptions = [
    "y = β₀ + β₁x",
    "y = β₀ + β₁x + β₂√x",
    "y = β₀ + β₁x + β₂x²",
    "y = β₀ + β₁x + β₂ln(x)",
    "y = β₀ + β₁x + β₂√(x+10)",
    "y = β₀ + β₁x + β₂ln(x) + β₃x³",
    "y = β₀ + β₁x + β₂cos(x)",
    "y = β₀ + β₁x² + β₂e^x",
    "y = β₀ + β₁x + β₂sin(x)",
    "y = β₀ + β₁x + β₂sin(x) + β₃x²",
    "y = β₀ + β₁x² + β₂tan(x) + β₃e^x",
    "y = β₀ + β₁x² + β₂cos(x) + β₃sin(x)"
]

results = []

# Créer une figure pour tous les graphiques
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.ravel()

print("\n" + "="*50)
print("ANALYSE DES 12 MODÈLES")
print("="*50)

for i in range(1, 13):
    try:
        # Créer la matrice de design
        X = create_design_matrix(xi, i)
        p = X.shape[1] - 1  # Nombre de variables explicatives (sans l'intercept)
        
        # Ajustement du modèle avec sklearn pour plus de stabilité
        model = LinearRegression()
        model.fit(X, yi)
        
        # Prédictions
        y_pred = model.predict(X)
        
        # Calcul du R²
        r2 = r2_score(yi, y_pred)
        
        # Calcul du R² ajusté
        adj_r2 = adjusted_r2(r2, len(yi), p)
        
        # Récupération des coefficients (avec intercept)
        beta = np.concatenate([[model.intercept_], model.coef_[1:]])
        
        # Stockage des résultats
        results.append({
            'model': i,
            'description': model_descriptions[i-1],
            'p': p,
            'beta': beta,
            'r2': r2,
            'adj_r2': adj_r2,
            'y_pred': y_pred
        })
        
        # Graphique
        ax = axes[i-1]
        ax.scatter(xi, yi, color='blue', alpha=0.7, label='Données réelles')
        ax.scatter(xi, y_pred, color='red', alpha=0.7, label='Prédictions')
        ax.plot(xi, y_pred, color='red', alpha=0.5)
        ax.set_title(f'Model {i}: R²={r2:.4f}, R²adj={adj_r2:.4f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Affichage des résultats
        print(f"\n--- Modèle {i}: {model_descriptions[i-1]} ---")
        print(f"Nombre de variables explicatives (p): {p}")
        print(f"Coefficients β: {[f'{b:.4f}' for b in beta]}")
        print(f"R²: {r2:.4f}")
        print(f"R² ajusté: {adj_r2:.4f}")
        
    except Exception as e:
        print(f"Erreur dans le modèle {i}: {str(e)}")
        continue

plt.tight_layout()
plt.show()

# 5. Comparaison des modèles
print("\n" + "="*50)
print("COMPARAISON DES MODÈLES")
print("="*50)

# Trier par R² ajusté
results.sort(key=lambda x: x['adj_r2'], reverse=True)

print("\nClassement par R² ajusté (meilleur au moins bon):")
for i, result in enumerate(results):
    print(f"{i+1}. Modèle {result['model']}: {result['description']}")
    print(f"   R² ajusté = {result['adj_r2']:.4f}")

# 6. Modèle global
print("\n" + "="*50)
print("MODÈLE GLOBAL")
print("="*50)

# Créer la matrice pour le modèle global
# y = β₀ + β₁x + β₂√x + β₃√(x+10) + β₄x² + β₅x³ + β₆cos(x) + β₇sin(x) + β₈tan(x) + β₉ln(x) + β₁₀e^x
X_global = np.column_stack([
    np.ones(len(xi)),     # β₀
    xi,                   # β₁x
    np.sqrt(xi),          # β₂√x
    np.sqrt(xi + 10),     # β₃√(x+10)
    xi**2,                # β₄x²
    xi**3,                # β₅x³
    np.cos(xi),           # β₆cos(x)
    np.sin(xi),           # β₇sin(x)
    np.tan(xi),           # β₈tan(x)
    np.log(xi),           # β₉ln(x)
    np.exp(xi)            # β₁₀e^x
])

# Ajustement du modèle global
model_global = LinearRegression()
model_global.fit(X_global, yi)

# Prédictions
y_pred_global = model_global.predict(X_global)

# Métriques
r2_global = r2_score(yi, y_pred_global)
p_global = X_global.shape[1] - 1
adj_r2_global = adjusted_r2(r2_global, len(yi), p_global)

print(f"Modèle global:")
print(f"R² = {r2_global:.4f}")
print(f"R² ajusté = {adj_r2_global:.4f}")
print(f"Nombre de variables: {p_global}")

# Graphique du modèle global
plt.figure(figsize=(10, 6))
plt.scatter(xi, yi, color='blue', alpha=0.7, label='Données réelles')
plt.scatter(xi, y_pred_global, color='red', alpha=0.7, label='Prédictions')
plt.plot(xi, y_pred_global, color='red', alpha=0.5)
plt.title(f'Modèle Global: R²={r2_global:.4f}, R²adj={adj_r2_global:.4f}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 7. Sélection de variables
print("\n" + "="*50)
print("SÉLECTION DE VARIABLES")
print("="*50)

# Noms des variables pour le modèle global
variable_names = ['intercept', 'x', '√x', '√(x+10)', 'x²', 'x³', 'cos(x)', 'sin(x)', 'tan(x)', 'ln(x)', 'e^x']

# Coefficients du modèle global
coefs_global = np.concatenate([[model_global.intercept_], model_global.coef_[1:]])

print("Coefficients du modèle global:")
for i, (name, coef) in enumerate(zip(variable_names, coefs_global)):
    print(f"{name}: {coef:.6f}")

# Fonction pour sélection ascendante
def forward_selection(X, y, variable_names, max_vars=None):
    if max_vars is None:
        max_vars = X.shape[1] - 1
    
    available_vars = list(range(1, X.shape[1]))  # Exclure l'intercept
    selected_vars = []
    best_adj_r2 = -np.inf
    
    print(f"\nSélection ascendante (max {max_vars} variables):")
    
    while len(selected_vars) < max_vars and available_vars:
        best_var = None
        best_score = -np.inf
        
        for var in available_vars:
            current_vars = [0] + selected_vars + [var]  # Toujours inclure l'intercept
            X_subset = X[:, current_vars]
            
            model_temp = LinearRegression()
            model_temp.fit(X_subset, y)
            y_pred_temp = model_temp.predict(X_subset)
            
            r2_temp = r2_score(y, y_pred_temp)
            adj_r2_temp = adjusted_r2(r2_temp, len(y), len(current_vars) - 1)
            
            if adj_r2_temp > best_score:
                best_score = adj_r2_temp
                best_var = var
        
        if best_var is not None and best_score > best_adj_r2:
            selected_vars.append(best_var)
            available_vars.remove(best_var)
            best_adj_r2 = best_score
            
            print(f"Step {len(selected_vars)}: Added {variable_names[best_var]}")
            print(f"  Variables: {[variable_names[v] for v in selected_vars]}")
            print(f"  R² ajusté: {best_adj_r2:.4f}")
        else:
            break
    
    return selected_vars, best_adj_r2

# Sélection ascendante pour les meilleurs modèles à 2, 3, et 4 variables
print("\nMeilleurs modèles emboîtés dans le modèle global:")
for n_vars in [2, 3, 4]:
    selected_vars, best_score = forward_selection(X_global, yi, variable_names, n_vars)
    print(f"\nMeilleur modèle à {n_vars} variables:")
    print(f"Variables: {[variable_names[v] for v in selected_vars]}")
    print(f"R² ajusté: {best_score:.4f}")

# Sélection ascendante complète
print("\nSélection ascendante complète:")
selected_vars_full, best_score_full = forward_selection(X_global, yi, variable_names)
print(f"\nMeilleur modèle (sélection ascendante complète):")
print(f"Variables: {[variable_names[v] for v in selected_vars_full]}")
print(f"R² ajusté: {best_score_full:.4f}")

# 8. Critère AIC
print("\n" + "="*50)
print("SÉLECTION PAR CRITÈRE AIC")
print("="*50)

# Fonction pour calculer l'AIC
def calculate_aic(y, y_pred, p):
    n = len(y)
    mse = np.mean((y - y_pred)**2)
    aic = n * np.log(mse) + 2 * (p + 1)  # +1 pour l'intercept
    return aic

# Sélection ascendante avec AIC
def forward_selection_aic(X, y, variable_names, max_vars=None):
    if max_vars is None:
        max_vars = X.shape[1] - 1
    
    available_vars = list(range(1, X.shape[1]))  # Exclure l'intercept
    selected_vars = []
    
    print(f"\nSélection ascendante avec AIC:")
    
    while len(selected_vars) < max_vars and available_vars:
        best_var = None
        best_aic = np.inf
        
        for var in available_vars:
            current_vars = [0] + selected_vars + [var]  # Toujours inclure l'intercept
            X_subset = X[:, current_vars]
            
            model_temp = LinearRegression()
            model_temp.fit(X_subset, y)
            y_pred_temp = model_temp.predict(X_subset)
            
            aic = calculate_aic(y, y_pred_temp, len(current_vars) - 1)
            
            if aic < best_aic:
                best_aic = aic
                best_var = var
        
        if best_var is not None:
            # Vérifier si l'ajout améliore l'AIC
            current_vars = [0] + selected_vars
            if current_vars != [0]:  # Si ce n'est pas le premier ajout
                X_current = X[:, current_vars]
                model_current = LinearRegression()
                model_current.fit(X_current, y)
                y_pred_current = model_current.predict(X_current)
                current_aic = calculate_aic(y, y_pred_current, len(current_vars) - 1)
                
                if best_aic >= current_aic:
                    break
            
            selected_vars.append(best_var)
            available_vars.remove(best_var)
            
            print(f"Step {len(selected_vars)}: Added {variable_names[best_var]}")
            print(f"  Variables: {[variable_names[v] for v in selected_vars]}")
            print(f"  AIC: {best_aic:.4f}")
        else:
            break
    
    return selected_vars, best_aic

# Sélection avec AIC
selected_vars_aic, best_aic = forward_selection_aic(X_global, yi, variable_names)
print(f"\nMeilleur modèle (sélection AIC):")
print(f"Variables: {[variable_names[v] for v in selected_vars_aic]}")
print(f"AIC: {best_aic:.4f}")

# Calcul du R² ajusté pour le modèle sélectionné par AIC
current_vars = [0] + selected_vars_aic
X_aic = X_global[:, current_vars]
model_aic = LinearRegression()
model_aic.fit(X_aic, yi)
y_pred_aic = model_aic.predict(X_aic)
r2_aic = r2_score(yi, y_pred_aic)
adj_r2_aic = adjusted_r2(r2_aic, len(yi), len(selected_vars_aic))
print(f"R² ajusté: {adj_r2_aic:.4f}")

print("\n" + "="*50)
print("RÉSUMÉ FINAL")
print("="*50)

print(f"Meilleur modèle individuel (1-12): Modèle {results[0]['model']}")
print(f"  Description: {results[0]['description']}")
print(f"  R² ajusté: {results[0]['adj_r2']:.4f}")

print(f"\nModèle global complet:")
print(f"  R² ajusté: {adj_r2_global:.4f}")

print(f"\nMeilleur modèle (sélection ascendante R² ajusté):")
print(f"  Variables: {[variable_names[v] for v in selected_vars_full]}")
print(f"  R² ajusté: {best_score_full:.4f}")

print(f"\nMeilleur modèle (sélection AIC):")
print(f"  Variables: {[variable_names[v] for v in selected_vars_aic]}")
print(f"  R² ajusté: {adj_r2_aic:.4f}")
print(f"  AIC: {best_aic:.4f}")