import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration pour de meilleures visualisations
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Couleurs Airbnb
colors = ["#FF5A5F", "#00A699", "#FC642D", "#484848", "#767676"]
sns.set_palette(sns.color_palette(colors))

# Charger les données
df = pd.read_csv("Data/Clean/train_partial_clean_standart.csv")

# Informations générales
print(f"Dimensions du jeu de données : {df.shape}")
print(f"Nombre d'appartements : {df.shape[0]}")
print(f"Nombre de caractéristiques : {df.shape[1]}")

# Aperçu des données
print("\nAperçu des 5 premières lignes :")
print(df.head())

# Types de données
print("\nTypes de données :")
print(df.dtypes)

# Statistiques descriptives
print("\nStatistiques descriptives :")
print(df.describe())

# Vérifier les valeurs manquantes
print("\nValeurs manquantes par colonne :")
print(df.isnull().sum())