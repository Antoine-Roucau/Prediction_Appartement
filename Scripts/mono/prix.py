import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Configuration des visualisations
plt.style.use('ggplot')
plt.rcParams['font.size'] = 12
colors = ["#FF5A5F", "#00A699", "#FC642D", "#484848", "#767676"]

def analyze_price_variables(df, output_dir):
    """
    Analyse univariée simplifiée des variables log_price et price
    avec enregistrement des visualisations
    
    Args:
        df: DataFrame contenant les données
        output_dir: Dossier de sortie pour sauvegarder les plots
    """
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculer price à partir de log_price
    df['price'] = np.exp(df['log_price'])
    
    # 1. Histogrammes
    # Histogramme de log_price
    plt.figure(figsize=(10, 6))
    sns.histplot(df['log_price'], kde=True, color=colors[0])
    plt.title('Distribution de log_price', fontsize=14)
    plt.xlabel('log_price', fontsize=12)
    plt.ylabel('Fréquence', fontsize=12)
    plt.axvline(df['log_price'].mean(), color=colors[2], linestyle='--', 
               label=f'Moyenne: {df["log_price"].mean():.2f}')
    plt.axvline(df['log_price'].median(), color=colors[1], linestyle='-', 
               label=f'Médiane: {df["log_price"].median():.2f}')
    plt.legend()
    plt.savefig(f"{output_dir}/histogramme_log_price.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Histogramme de price
    plt.figure(figsize=(10, 6))
    max_price = df['price']
    sns.histplot(df[df['price'] <= max_price]['price'], kde=True, color=colors[0])
    plt.title('Distribution du price', fontsize=14)
    plt.xlabel('Price', fontsize=12)
    plt.ylabel('Fréquence', fontsize=12)
    plt.axvline(df[df['price'] <= max_price]['price'].mean(), color=colors[2], linestyle='--', 
               label=f'Moyenne: ${df[df["price"] <= max_price]["price"].mean():.2f}')
    plt.axvline(df[df['price'] <= max_price]['price'].median(), color=colors[1], linestyle='-', 
               label=f'Médiane: ${df[df["price"] <= max_price]["price"].median():.2f}')
    plt.legend()
    plt.savefig(f"{output_dir}/histogramme_price.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Boxplots
    # Boxplot de log_price
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df['log_price'], color=colors[0])
    plt.title('Boxplot de log_price', fontsize=14)
    plt.ylabel('log_price', fontsize=12)
    plt.savefig(f"{output_dir}/boxplot_log_price.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Boxplot de price (tronqué)
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df[df['price'] <= max_price]['price'], color=colors[0])
    plt.title('Boxplot du prix (99% des données)', fontsize=14)
    plt.ylabel('Prix ($)', fontsize=12)
    plt.savefig(f"{output_dir}/boxplot_price.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. QQ-Plots pour vérifier la normalité
    # QQ-Plot de log_price
    plt.figure(figsize=(10, 6))
    stats.probplot(df['log_price'], plot=plt)
    plt.title('QQ-Plot de log_price', fontsize=14)
    plt.savefig(f"{output_dir}/qqplot_log_price.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # QQ-Plot de price (tronqué)
    plt.figure(figsize=(10, 6))
    stats.probplot(df[df['price'] <= max_price]['price'], plot=plt)
    plt.title('QQ-Plot du price', fontsize=14)
    plt.savefig(f"{output_dir}/qqplot_price.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Graphiques de densité
    # Densité de log_price
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df['log_price'], fill=True, color=colors[0])
    plt.title('Densité de log_price', fontsize=14)
    plt.xlabel('log_price', fontsize=12)
    plt.ylabel('Densité', fontsize=12)
    plt.savefig(f"{output_dir}/densite_log_price.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Densité de price (tronqué)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df[df['price'] <= max_price]['price'], fill=True, color=colors[0])
    plt.title('Densité du prix (99% des données)', fontsize=14)
    plt.xlabel('Prix ($)', fontsize=12)
    plt.ylabel('Densité', fontsize=12)
    plt.savefig(f"{output_dir}/densite_price.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Statistiques descriptives
    # Calculer les statistiques
    stats_log = df['log_price'].describe()
    stats_real = df['price'].describe()
    
    stats_log['skewness'] = df['log_price'].skew()
    stats_log['kurtosis'] = df['log_price'].kurtosis()
    stats_real['skewness'] = df['price'].skew()
    stats_real['kurtosis'] = df['price'].kurtosis()
    
    # Formater les statistiques pour affichage
    stats_table = pd.DataFrame({
        'Statistique': [
            'Moyenne', 
            'Écart-type', 
            'Minimum', 
            '25%', 
            'Médiane (50%)', 
            '75%', 
            'Maximum',
            'Asymétrie (Skewness)',
            'Aplatissement (Kurtosis)'
        ],
        'log_price': [
            f"{stats_log['mean']:.4f}",
            f"{stats_log['std']:.4f}",
            f"{stats_log['min']:.4f}",
            f"{stats_log['25%']:.4f}",
            f"{stats_log['50%']:.4f}",
            f"{stats_log['75%']:.4f}",
            f"{stats_log['max']:.4f}",
            f"{stats_log['skewness']:.4f}",
            f"{stats_log['kurtosis']:.4f}"
        ],
        'price ($)': [
            f"{stats_real['mean']:.2f}",
            f"{stats_real['std']:.2f}",
            f"{stats_real['min']:.2f}",
            f"{stats_real['25%']:.2f}",
            f"{stats_real['50%']:.2f}",
            f"{stats_real['75%']:.2f}",
            f"{stats_real['max']:.2f}",
            f"{stats_real['skewness']:.2f}",
            f"{stats_real['kurtosis']:.2f}"
        ]
    })
    
    # Sauvegarder les statistiques dans un fichier CSV
    stats_table.to_csv(f"{output_dir}/statistiques_prix.csv", index=False)
    
    # Créer un graphique avec le tableau des statistiques
    plt.figure(figsize=(10, 8))
    plt.axis('off')
    table = plt.table(
        cellText=stats_table.values,
        colLabels=stats_table.columns,
        cellLoc='center',
        loc='center',
        bbox=[0.0, 0.0, 1.0, 1.0]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    plt.title('Statistiques descriptives des prix', fontsize=16, pad=20)
    plt.savefig(f"{output_dir}/tableau_statistiques.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Afficher quelques statistiques dans la console
    print("\nAnalyse univariée des prix terminée!")
    print(f"8 visualisations ont été enregistrées dans le dossier: {output_dir}")
    print(f"\nPrix médian: ${df['price'].median():.2f}")
    print(f"Prix moyen: ${df['price'].mean():.2f}")
    print(f"Intervalle de prix (99% des données): ${df['price'].quantile(0.01):.2f} - ${df['price'].quantile(0.99):.2f}")
    
    return stats_table

# Utilisation de la fonction
file_path = "Data/Clean/train_partial_clean_standart.csv"
output_dir = "Data/Visual/plot/Mono/prix"
df = pd.read_csv(file_path)
stats_table = analyze_price_variables(df, output_dir)

# Afficher le tableau de statistiques
print("\nTableau des statistiques descriptives:")
print(stats_table)