"""
Script d'analyse exploratoire des données pour le projet de prédiction de prix d'appartements Airbnb.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re

# Configuration pour de meilleures visualisations
try:
    plt.style.use('seaborn-v0_8-whitegrid')  # Pour les versions récentes de matplotlib
except:
    try:
        plt.style.use('seaborn-whitegrid')  # Pour les anciennes versions
    except:
        plt.style.use('ggplot')  # Alternative sûre

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Créer une palette de couleurs personnalisée inspirée d'Airbnb
colors = ["#FF5A5F", "#00A699", "#FC642D", "#484848", "#767676"]
airbnb_palette = sns.color_palette(colors)
sns.set_palette(airbnb_palette)

def count_amenities(amenities_str):
    """
    Fonction robuste pour compter les aménités, sans dépendre strictement de json.loads
    """
    try:
        # Supprimer les accolades au début et à la fin
        if amenities_str.startswith('{') and amenities_str.endswith('}'):
            amenities_str = amenities_str[1:-1]
        
        # Utiliser une expression régulière pour capturer les éléments entre virgules
        amenities_list = re.findall(r'"([^"]*)"', amenities_str)
        return len(amenities_list)
    except:
        # En cas d'erreur, retourner 0
        return 0

def extract_amenities_list(amenities_str):
    """
    Fonction pour extraire la liste des aménités de manière robuste
    """
    try:
        # Supprimer les accolades au début et à la fin
        if amenities_str.startswith('{') and amenities_str.endswith('}'):
            amenities_str = amenities_str[1:-1]
        
        # Utiliser une expression régulière pour capturer les éléments entre guillemets
        amenities_list = re.findall(r'"([^"]*)"', amenities_str)
        return amenities_list
    except:
        return []

def load_and_describe_data(file_path):
    """
    Charge le jeu de données et fournit une description détaillée
    """
    # Chargement des données
    print(f"Chargement des données depuis {file_path}...")
    df = pd.read_csv(file_path)
    
    # Informations générales sur le dataset
    print(f"Dimensions du jeu de données : {df.shape}")
    print(f"Nombre d'appartements : {df.shape[0]}")
    print(f"Nombre de caractéristiques : {df.shape[1]}")
    
    # Convertir log_price en prix réel pour une meilleure interprétation
    df['price'] = np.exp(df['log_price'])
    
    # Types de données
    print("\nTypes de données :")
    print(df.dtypes)
    
    # Conversion des dates si nécessaire
    if 'host_since' in df.columns:
        print("Conversion des dates host_since...")
        df['host_since'] = pd.to_datetime(df['host_since'])
        df['host_experience_years'] = (pd.to_datetime('today') - df['host_since']).dt.days / 365.25
    
    # Extraction du nombre d'aménités par propriété - version robuste
    print("Extraction du nombre d'aménités...")
    df['amenities_count'] = df['amenities'].apply(count_amenities)
    
    return df

def generate_summary_statistics(df):
    """
    Génère des statistiques descriptives pour le jeu de données
    """
    # Résumé statistique des variables numériques
    num_vars = df.select_dtypes(include=['number']).columns
    print("\nStatistiques descriptives des variables numériques :")
    print(df[num_vars].describe().T)
    
    # Distribution des variables catégorielles
    cat_vars = df.select_dtypes(include=['object']).columns
    print("\nDistribution des variables catégorielles :")
    
    for col in cat_vars:
        if col not in ['id', 'amenities', 'neighbourhood']:  # Exclure les colonnes à forte cardinalité
            value_counts = df[col].value_counts()
            print(f"\n{col}:")
            print(value_counts)
            print(f"Nombre de valeurs uniques: {len(value_counts)}")
    
    return

def price_distribution_analysis(df):
    """
    Analyse la distribution des prix et génère des visualisations
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Distribution des prix logarithmiques
    sns.histplot(data=df, x='log_price', kde=True, ax=axes[0])
    axes[0].set_title('Distribution des prix (échelle logarithmique)', fontsize=14)
    axes[0].set_xlabel('Log(Prix)', fontsize=12)
    axes[0].set_ylabel('Fréquence', fontsize=12)
    
    # Distribution des prix réels (avec troncature pour les valeurs extrêmes)
    max_price = df['price'].quantile(0.99)  # Tronquer à 99% pour éviter les valeurs extrêmes
    sns.histplot(data=df[df['price'] <= max_price], x='price', kde=True, ax=axes[1])
    axes[1].set_title('Distribution des prix (échelle réelle, 99% des données)', fontsize=14)
    axes[1].set_xlabel('Prix ($)', fontsize=12)
    axes[1].set_ylabel('Fréquence', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # Statistiques clés sur les prix
    print("\nStatistiques des prix :")
    price_stats = df['price'].describe().to_dict()
    for key, value in price_stats.items():
        if key != 'count':
            print(f"{key.capitalize()}: ${value:.2f}")
    
    print(f"\nPrix médian global: ${df['price'].median():.2f}")
    print(f"Prix moyen global: ${df['price'].mean():.2f}")
    
    # Prix par ville
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='city', y='price', data=df[df['price'] <= max_price])
    plt.title('Distribution des prix par ville', fontsize=16)
    plt.xlabel('Ville', fontsize=14)
    plt.ylabel('Prix ($)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Prix médian par ville
    city_median = df.groupby('city')['price'].median().sort_values(ascending=False)
    
    plt.figure(figsize=(14, 8))
    city_median.plot(kind='bar', color=colors[0])
    plt.title('Prix médian par ville', fontsize=16)
    plt.xlabel('Ville', fontsize=14)
    plt.ylabel('Prix médian ($)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return

def property_characteristics_analysis(df):
    """
    Analyse les caractéristiques des propriétés et leur relation avec le prix
    """
    # Sélectionner les caractéristiques numériques pertinentes
    features = ['accommodates', 'bedrooms', 'beds', 'bathrooms', 'price']
    
    # Matrice de corrélation
    plt.figure(figsize=(10, 8))
    corr_matrix = df[features].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', 
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Matrice de corrélation des caractéristiques', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Relation entre les caractéristiques des propriétés et le prix
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(features[:-1]):
        sns.regplot(x=feature, y='price', data=df, 
                   scatter_kws={'alpha':0.4, 's':10}, line_kws={'color':colors[0]}, ax=axes[i])
        axes[i].set_title(f'Relation entre {feature} et Prix', fontsize=14)
        axes[i].set_xlabel(feature, fontsize=12)
        axes[i].set_ylabel('Prix ($)', fontsize=12)
        # Limiter les y pour une meilleure lisibilité
        axes[i].set_ylim(0, df['price'].quantile(0.95))
    
    plt.tight_layout()
    plt.show()
    
    # Distribution des types de propriétés
    plt.figure(figsize=(14, 8))
    property_counts = df['property_type'].value_counts()
    property_counts.plot(kind='bar', color=colors[1])
    plt.title('Distribution des types de propriétés', fontsize=16)
    plt.xlabel('Type de propriété', fontsize=14)
    plt.ylabel('Nombre', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Prix médian par type de propriété
    plt.figure(figsize=(14, 8))
    property_median = df.groupby('property_type')['price'].median().sort_values(ascending=False)
    property_median.plot(kind='bar', color=colors[2])
    plt.title('Prix médian par type de propriété', fontsize=16)
    plt.xlabel('Type de propriété', fontsize=14)
    plt.ylabel('Prix médian ($)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Distribution des types de chambre
    plt.figure(figsize=(14, 8))
    room_counts = df['room_type'].value_counts()
    room_counts.plot(kind='bar', color=colors[3])
    plt.title('Distribution des types de chambre', fontsize=16)
    plt.xlabel('Type de chambre', fontsize=14)
    plt.ylabel('Nombre', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Prix médian par type de chambre
    plt.figure(figsize=(14, 8))
    room_median = df.groupby('room_type')['price'].median().sort_values(ascending=False)
    room_median.plot(kind='bar', color=colors[4])
    plt.title('Prix médian par type de chambre', fontsize=16)
    plt.xlabel('Type de chambre', fontsize=14)
    plt.ylabel('Prix médian ($)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return

def location_analysis(df):
    """
    Analyse l'impact de la localisation sur les prix
    """
    # Nombre de propriétés par ville
    plt.figure(figsize=(14, 8))
    city_counts = df['city'].value_counts()
    city_counts.plot(kind='bar', color=colors[0])
    plt.title('Nombre de propriétés par ville', fontsize=16)
    plt.xlabel('Ville', fontsize=14)
    plt.ylabel('Nombre de propriétés', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Visualisation géographique des prix (scatter plot)
    # Prenons NYC comme exemple
    if len(df[df['city'] == 'NYC']) > 0:
        nyc_df = df[df['city'] == 'NYC'].copy()
        
        # Normaliser les prix pour la couleur
        max_price = nyc_df['price'].max()
        nyc_df['price_normalized'] = nyc_df['price'] / max_price
        
        plt.figure(figsize=(12, 10))
        plt.scatter(
            nyc_df['longitude'], 
            nyc_df['latitude'],
            c=nyc_df['price'],
            cmap='coolwarm',
            alpha=0.7,
            s=20
        )
        plt.colorbar(label='Prix ($)')
        plt.title('Distribution géographique des prix à New York', fontsize=16)
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        # Analyse des prix par quartier à NYC
        if 'neighbourhood' in nyc_df.columns:
            # Prendre les 15 quartiers les plus représentés
            top_neighborhoods = nyc_df['neighbourhood'].value_counts().head(15).index
            neighborhood_data = nyc_df[nyc_df['neighbourhood'].isin(top_neighborhoods)]
            
            plt.figure(figsize=(16, 10))
            sns.boxplot(x='neighbourhood', y='price', data=neighborhood_data)
            plt.title('Prix par quartier à New York (Top 15)', fontsize=16)
            plt.xlabel('Quartier', fontsize=14)
            plt.ylabel('Prix ($)', fontsize=14)
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()
    
    # Faire la même chose pour SF si disponible
    if len(df[df['city'] == 'SF']) > 0:
        sf_df = df[df['city'] == 'SF'].copy()
        
        plt.figure(figsize=(12, 10))
        plt.scatter(
            sf_df['longitude'], 
            sf_df['latitude'],
            c=sf_df['price'],
            cmap='coolwarm',
            alpha=0.7,
            s=20
        )
        plt.colorbar(label='Prix ($)')
        plt.title('Distribution géographique des prix à San Francisco', fontsize=16)
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    return

def amenities_analysis(df):
    """
    Analyse des aménités et leur impact sur les prix
    """
    # Extraire et compter les aménités les plus courantes avec notre méthode robuste
    all_amenities = []
    for amenities_str in df['amenities']:
        amenities_list = extract_amenities_list(amenities_str)
        all_amenities.extend(amenities_list)
    
    # Compter les occurrences
    amenities_count = pd.Series(all_amenities).value_counts()
    
    # Afficher les 20 aménités les plus courantes
    plt.figure(figsize=(16, 10))
    amenities_count.head(20).plot(kind='bar', color=colors[1])
    plt.title('Les 20 aménités les plus courantes', fontsize=16)
    plt.xlabel('Aménité', fontsize=14)
    plt.ylabel('Nombre d\'occurrences', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Analyser l'impact du nombre d'aménités sur le prix
    plt.figure(figsize=(14, 8))
    sns.regplot(x='amenities_count', y='price', data=df, 
               scatter_kws={'alpha':0.4, 's':10}, line_kws={'color':colors[0]})
    plt.title('Relation entre le nombre d\'aménités et le prix', fontsize=16)
    plt.xlabel('Nombre d\'aménités', fontsize=14)
    plt.ylabel('Prix ($)', fontsize=14)
    plt.ylim(0, df['price'].quantile(0.95))  # Limiter pour une meilleure lisibilité
    plt.tight_layout()
    plt.show()
    
    # Analyse de certaines aménités spécifiques
    # Sélectionner les 10 aménités les plus fréquentes
    top_amenities = amenities_count.head(10).index.tolist()
    
    # Créer des colonnes binaires pour ces aménités
    for amenity in top_amenities:
        amenity_clean = amenity.replace(" ", "_").lower()
        df[f'has_{amenity_clean}'] = df['amenities'].apply(
            lambda x: 1 if amenity in x else 0
        )
    
    # Comparer le prix médian avec/sans ces aménités
    amenity_impact = {}
    for amenity in top_amenities:
        amenity_clean = amenity.replace(" ", "_").lower()
        col_name = f'has_{amenity_clean}'
        with_amenity = df[df[col_name] == 1]['price'].median()
        without_amenity = df[df[col_name] == 0]['price'].median()
        amenity_impact[amenity] = {
            'with': with_amenity,
            'without': without_amenity,
            'diff_pct': (with_amenity - without_amenity) / without_amenity * 100 if without_amenity > 0 else 0
        }
    
    # Convertir en dataframe pour visualisation
    impact_df = pd.DataFrame(amenity_impact).T
    impact_df = impact_df.sort_values('diff_pct', ascending=False)
    
    plt.figure(figsize=(16, 8))
    impact_df['diff_pct'].plot(kind='bar', color=colors[2])
    plt.title('Impact des aménités sur le prix (% de différence)', fontsize=16)
    plt.xlabel('Aménité', fontsize=14)
    plt.ylabel('Différence de prix en %', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return

def host_analysis(df):
    """
    Analyse de l'influence des hôtes sur les prix
    """
    if 'host_experience_years' in df.columns:
        # Relation entre l'expérience de l'hôte et le prix
        plt.figure(figsize=(14, 8))
        sns.regplot(x='host_experience_years', y='price', data=df, 
                   scatter_kws={'alpha':0.4, 's':10}, line_kws={'color':colors[2]})
        plt.title('Relation entre l\'expérience de l\'hôte et le prix', fontsize=16)
        plt.xlabel('Années d\'expérience', fontsize=14)
        plt.ylabel('Prix ($)', fontsize=14)
        plt.ylim(0, df['price'].quantile(0.95))  # Limiter pour une meilleure lisibilité
        plt.tight_layout()
        plt.show()
        
        # Statistiques d'expérience des hôtes
        exp_stats = df['host_experience_years'].describe()
        print("\nStatistiques sur l'expérience des hôtes (années):")
        for stat, value in exp_stats.items():
            print(f"{stat}: {value:.2f}")
    
    # Impact de l'identité vérifiée sur le prix
    if 'host_identity_verified' in df.columns:
        # Créer un barplot pour comparer le prix médian
        verified_median = df[df['host_identity_verified'] == 't']['price'].median()
        not_verified_median = df[df['host_identity_verified'] == 'f']['price'].median()
        
        plt.figure(figsize=(10, 6))
        plt.bar(['Non vérifié (f)', 'Vérifié (t)'], 
                [not_verified_median, verified_median],
                color=[colors[3], colors[0]])
        plt.title('Prix médian selon la vérification d\'identité', fontsize=16)
        plt.xlabel('Statut de vérification d\'identité', fontsize=14)
        plt.ylabel('Prix médian ($)', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Boxplot pour montrer la distribution
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='host_identity_verified', y='price', 
                   data=df[df['price'] <= df['price'].quantile(0.95)])
        plt.title('Distribution des prix selon la vérification d\'identité', fontsize=16)
        plt.xlabel('Identité vérifiée', fontsize=14)
        plt.ylabel('Prix ($)', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Statistiques sur l'identité vérifiée
        id_counts = df['host_identity_verified'].value_counts()
        id_pcts = df['host_identity_verified'].value_counts(normalize=True) * 100
        
        print("\nDistribution des hôtes par statut de vérification d'identité:")
        for status, count in id_counts.items():
            print(f"{status}: {count} hôtes ({id_pcts[status]:.2f}%)")
    
    return

def review_analysis(df):
    """
    Analyse de l'impact des avis sur les prix
    """
    # Statistiques descriptives sur les avis
    review_stats = df['number_of_reviews'].describe()
    print("\nStatistiques sur le nombre d'avis:")
    for stat, value in review_stats.items():
        print(f"{stat}: {value:.2f}")
    
    # Relation entre le nombre d'avis et le prix
    plt.figure(figsize=(14, 8))
    sns.regplot(x='number_of_reviews', y='price', data=df,
               scatter_kws={'alpha':0.4, 's':10}, line_kws={'color':colors[3]})
    plt.title('Relation entre le nombre d\'avis et le prix', fontsize=16)
    plt.xlabel('Nombre d\'avis', fontsize=14)
    plt.ylabel('Prix ($)', fontsize=14)
    plt.ylim(0, df['price'].quantile(0.95))  # Limiter pour une meilleure lisibilité
    plt.tight_layout()
    plt.show()
    
    # Distribution des nombres d'avis
    plt.figure(figsize=(14, 8))
    sns.histplot(df['number_of_reviews'], kde=True, bins=50)
    plt.title('Distribution du nombre d\'avis', fontsize=16)
    plt.xlabel('Nombre d\'avis', fontsize=14)
    plt.ylabel('Fréquence', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Relation entre la note d'évaluation et le prix
    if 'review_scores_rating' in df.columns:
        rating_stats = df['review_scores_rating'].describe()
        print("\nStatistiques sur les notes d'évaluation:")
        for stat, value in rating_stats.items():
            if not pd.isna(value):
                print(f"{stat}: {value:.2f}")
        
        plt.figure(figsize=(14, 8))
        sns.regplot(x='review_scores_rating', y='price', 
                   data=df.dropna(subset=['review_scores_rating']),
                   scatter_kws={'alpha':0.4, 's':10}, line_kws={'color':colors[4]})
        plt.title('Relation entre la note d\'évaluation et le prix', fontsize=16)
        plt.xlabel('Note d\'évaluation', fontsize=14)
        plt.ylabel('Prix ($)', fontsize=14)
        plt.ylim(0, df['price'].quantile(0.95))  # Limiter pour une meilleure lisibilité
        plt.tight_layout()
        plt.show()
        
        # Distribution des notes d'évaluation
        plt.figure(figsize=(14, 8))
        sns.histplot(df['review_scores_rating'].dropna(), kde=True, bins=50)
        plt.title('Distribution des notes d\'évaluation', fontsize=16)
        plt.xlabel('Note d\'évaluation', fontsize=14)
        plt.ylabel('Fréquence', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Créer des catégories de notation
        df['rating_category'] = pd.cut(
            df['review_scores_rating'],
            bins=[0, 70, 80, 90, 100],
            labels=['< 70', '70-80', '80-90', '90-100']
        )
        
        # Prix médian par catégorie de notation
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='rating_category', y='price', 
                   data=df.dropna(subset=['rating_category', 'price']))
        plt.title('Prix par catégorie de notes d\'évaluation', fontsize=16)
        plt.xlabel('Catégorie de note', fontsize=14)
        plt.ylabel('Prix ($)', fontsize=14)
        plt.ylim(0, df['price'].quantile(0.95))
        plt.tight_layout()
        plt.show()
    
    return

def run_full_eda(file_path):
    """
    Exécute l'analyse exploratoire complète
    """
    print("=== ANALYSE EXPLORATOIRE DES DONNÉES AIRBNB ===\n")
    
    # Charger et décrire les données
    df = load_and_describe_data(file_path)
    
    # Générer les statistiques descriptives
    generate_summary_statistics(df)
    
    # Analyse des prix
    print("\n--- ANALYSE DES PRIX ---")
    price_distribution_analysis(df)
    
    # Analyse des caractéristiques des propriétés
    print("\n--- ANALYSE DES CARACTÉRISTIQUES DES PROPRIÉTÉS ---")
    property_characteristics_analysis(df)
    
    # Analyse de la localisation
    print("\n--- ANALYSE DE LA LOCALISATION ---")
    location_analysis(df)
    
    # Analyse des aménités
    print("\n--- ANALYSE DES AMÉNITÉS ---")
    amenities_analysis(df)
    
    # Analyse des hôtes
    print("\n--- ANALYSE DES HÔTES ---")
    host_analysis(df)
    
    # Analyse des avis
    print("\n--- ANALYSE DES AVIS ---")
    review_analysis(df)
    
    print("\nAnalyse exploratoire terminée !")
    return df

# Appeler la fonction principale avec le chemin du fichier nettoyé
if __name__ == "__main__":
    file_path = "Data/Clean/train_partial_clean_standart.csv"
    df_analyzed = run_full_eda(file_path)