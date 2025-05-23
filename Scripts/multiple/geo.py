import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# Configuration pour de meilleures visualisations
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Couleurs Airbnb
colors = ["#FF5A5F", "#00A699", "#FC642D", "#484848", "#767676"]
sns.set_palette(sns.color_palette(colors))

def analyze_location_impact(file_path, output_dir="Data/Visual/plot/Multiple/localisation"):
    """
    Analyse approfondie de l'impact de la localisation sur les prix Airbnb
    avec sauvegarde des graphiques
    """
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Chargement des données depuis {file_path}...")
    df = pd.read_csv(file_path)
    
    # Initialiser un compteur pour les figures
    fig_count = 1
    
    # Convertir log_price en price pour certaines visualisations
    df['price'] = np.exp(df['log_price'])
    
    # Variables géographiques
    location_features = ['city', 'neighbourhood', 'latitude', 'longitude']
    
    # 1. Statistiques descriptives par ville
    print("\n=== Statistiques descriptives par ville ===")
    city_stats = df.groupby('city').agg({
        'price': ['count', 'mean', 'median', 'std'],
        'log_price': ['mean', 'median', 'std']
    }).round(2)
    
    print(city_stats)
    
    # Sauvegarder les statistiques dans un CSV
    city_stats.to_csv(f"{output_dir}/01_stats_descriptives_par_ville.csv")
    
    # 2. Distribution des propriétés par ville
    print("\n=== Distribution des propriétés par ville ===")
    city_counts = df['city'].value_counts()
    print(city_counts)
    
    plt.figure(figsize=(12, 6))
    city_counts.plot(kind='bar', color=colors[0])
    plt.title('Nombre de propriétés par ville', fontsize=16)
    plt.xlabel('Ville', fontsize=14)
    plt.ylabel('Nombre de propriétés', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fig_count:02d}_distribution_proprietes_par_ville.png", dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    
    # 3. Prix médian par ville
    plt.figure(figsize=(12, 6))
    city_median = df.groupby('city')['price'].median().sort_values(ascending=False)
    city_median.plot(kind='bar', color=colors[1])
    plt.title('Prix médian par ville', fontsize=16)
    plt.xlabel('Ville', fontsize=14)
    plt.ylabel('Prix médian ($)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fig_count:02d}_prix_median_par_ville.png", dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    
    # 4. Boxplot des prix par ville
    plt.figure(figsize=(14, 8))
    # Limiter les prix pour une meilleure visualisation
    max_price = df['price'].quantile(0.95)
    sns.boxplot(x='city', y='price', data=df[df['price'] <= max_price])
    plt.title('Distribution des prix par ville (95% des données)', fontsize=16)
    plt.xlabel('Ville', fontsize=14)
    plt.ylabel('Prix ($)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fig_count:02d}_boxplot_prix_par_ville.png", dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    
    # 5. Analyse des quartiers (top 15 par ville)
    print("\n=== Analyse des quartiers par ville ===")
    
    # Analyser les quartiers pour chaque ville
    for city in df['city'].unique():
        city_data = df[df['city'] == city]
        
        # Top 15 quartiers par nombre de propriétés
        top_neighbourhoods = city_data['neighbourhood'].value_counts().head(15)
        
        if len(top_neighbourhoods) > 5:  # Seulement si assez de quartiers
            plt.figure(figsize=(16, 8))
            top_neighbourhoods.plot(kind='bar', color=colors[2])
            plt.title(f'Top 15 des quartiers par nombre de propriétés - {city}', fontsize=16)
            plt.xlabel('Quartier', fontsize=14)
            plt.ylabel('Nombre de propriétés', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{fig_count:02d}_quartiers_populaires_{city}.png", dpi=300, bbox_inches='tight')
            plt.close()
            fig_count += 1
            
            # Prix médian par quartier
            neighbourhood_median = city_data.groupby('neighbourhood')['price'].median().sort_values(ascending=False).head(15)
            
            plt.figure(figsize=(16, 8))
            neighbourhood_median.plot(kind='bar', color=colors[3])
            plt.title(f'Prix médian par quartier (Top 15) - {city}', fontsize=16)
            plt.xlabel('Quartier', fontsize=14)
            plt.ylabel('Prix médian ($)', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{fig_count:02d}_prix_median_quartiers_{city}.png", dpi=300, bbox_inches='tight')
            plt.close()
            fig_count += 1
    
    # 6. Analyse géographique par coordonnées GPS
    print("\n=== Analyse géographique par coordonnées GPS ===")
    
    # Vérifier la présence des coordonnées
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Statistiques des coordonnées
        coord_stats = df[['latitude', 'longitude']].describe()
        print("Statistiques des coordonnées:")
        print(coord_stats)
        
        # Sauvegarder les statistiques des coordonnées
        coord_stats.to_csv(f"{output_dir}/{fig_count:02d}_stats_coordonnees.csv")
        fig_count += 1
        
        # Scatter plot géographique coloré par prix
        plt.figure(figsize=(14, 10))
        scatter = plt.scatter(df['longitude'], df['latitude'], 
                            c=df['price'], cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Prix ($)')
        plt.title('Distribution géographique des prix', fontsize=16)
        plt.xlabel('Longitude', fontsize=14)
        plt.ylabel('Latitude', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fig_count:02d}_distribution_geographique_prix.png", dpi=300, bbox_inches='tight')
        plt.close()
        fig_count += 1
        
        # Analyse par ville avec coordonnées
        for city in df['city'].unique():
            city_data = df[df['city'] == city]
            
            if len(city_data) > 100:  # Seulement pour les villes avec assez de données
                plt.figure(figsize=(12, 10))
                scatter = plt.scatter(city_data['longitude'], city_data['latitude'],
                                    c=city_data['price'], cmap='coolwarm', alpha=0.7, s=30)
                plt.colorbar(scatter, label='Prix ($)')
                plt.title(f'Distribution géographique des prix - {city}', fontsize=16)
                plt.xlabel('Longitude', fontsize=12)
                plt.ylabel('Latitude', fontsize=12)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/{fig_count:02d}_geo_prix_{city}.png", dpi=300, bbox_inches='tight')
                plt.close()
                fig_count += 1
        
        # 7. Analyse de densité géographique
        print("\n=== Analyse de densité géographique ===")
        
        # Créer des bins pour la densité
        df['lat_bin'] = pd.cut(df['latitude'], bins=20, labels=False)
        df['lon_bin'] = pd.cut(df['longitude'], bins=20, labels=False)
        
        # Calculer la densité et le prix moyen par zone
        density_analysis = df.groupby(['lat_bin', 'lon_bin']).agg({
            'id': 'count',
            'price': 'mean',
            'log_price': 'mean'
        }).reset_index()
        density_analysis.columns = ['lat_bin', 'lon_bin', 'count', 'avg_price', 'avg_log_price']
        
        # Heatmap de la densité
        density_pivot = density_analysis.pivot(index='lat_bin', columns='lon_bin', values='count')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(density_pivot, cmap='YlOrRd', cbar_kws={'label': 'Nombre de propriétés'})
        plt.title('Densité géographique des propriétés Airbnb', fontsize=16)
        plt.xlabel('Longitude (bins)', fontsize=14)
        plt.ylabel('Latitude (bins)', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fig_count:02d}_heatmap_densite.png", dpi=300, bbox_inches='tight')
        plt.close()
        fig_count += 1
        
        # Heatmap des prix moyens
        price_pivot = density_analysis.pivot(index='lat_bin', columns='lon_bin', values='avg_price')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(price_pivot, cmap='viridis', cbar_kws={'label': 'Prix moyen ($)'})
        plt.title('Prix moyen par zone géographique', fontsize=16)
        plt.xlabel('Longitude (bins)', fontsize=14)
        plt.ylabel('Latitude (bins)', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fig_count:02d}_heatmap_prix_moyen.png", dpi=300, bbox_inches='tight')
        plt.close()
        fig_count += 1
    
    # 8. Analyse des corrélations géographiques
    print("\n=== Analyse des corrélations géographiques ===")
    
    # Calculer les corrélations avec les coordonnées
    geographic_vars = ['latitude', 'longitude']
    if all(var in df.columns for var in geographic_vars):
        geo_corr = df[geographic_vars + ['log_price', 'price']].corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(geo_corr, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.title('Corrélations géographiques avec les prix', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fig_count:02d}_correlation_geo_prix.png", dpi=300, bbox_inches='tight')
        plt.close()
        fig_count += 1
        
        # Sauvegarder la matrice de corrélation
        geo_corr.to_csv(f"{output_dir}/{fig_count:02d}_correlation_geo_prix.csv")
        fig_count += 1
    
    # 9. Analyse comparative des villes
    print("\n=== Analyse comparative des villes ===")
    
    # Statistiques comparatives par ville
    city_comparison = df.groupby('city').agg({
        'price': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'accommodates': 'mean',
        'bedrooms': 'mean',
        'bathrooms': 'mean'
    }).round(2)
    
    # Sauvegarder la comparaison
    city_comparison.to_csv(f"{output_dir}/{fig_count:02d}_comparaison_villes.csv")
    fig_count += 1
    
    # Graphique comparatif des caractéristiques moyennes par ville
    city_metrics = df.groupby('city').agg({
        'price': 'mean',
        'accommodates': 'mean',
        'bedrooms': 'mean',
        'bathrooms': 'mean'
    }).round(2)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Prix moyen
    city_metrics['price'].plot(kind='bar', ax=axes[0,0], color=colors[0])
    axes[0,0].set_title('Prix moyen par ville', fontsize=14)
    axes[0,0].set_ylabel('Prix ($)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Capacité d'accueil moyenne
    city_metrics['accommodates'].plot(kind='bar', ax=axes[0,1], color=colors[1])
    axes[0,1].set_title('Capacité d\'accueil moyenne par ville', fontsize=14)
    axes[0,1].set_ylabel('Nombre de personnes')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Nombre de chambres moyen
    city_metrics['bedrooms'].plot(kind='bar', ax=axes[1,0], color=colors[2])
    axes[1,0].set_title('Nombre de chambres moyen par ville', fontsize=14)
    axes[1,0].set_ylabel('Nombre de chambres')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Nombre de salles de bains moyen
    city_metrics['bathrooms'].plot(kind='bar', ax=axes[1,1], color=colors[3])
    axes[1,1].set_title('Nombre de salles de bains moyen par ville', fontsize=14)
    axes[1,1].set_ylabel('Nombre de salles de bains')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fig_count:02d}_comparaison_caracteristiques_villes.png", dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    
    # 10. Analyse de la variance des prix intra-ville
    print("\n=== Analyse de la variance des prix intra-ville ===")
    
    # Coefficient de variation par ville
    city_variation = df.groupby('city').agg({
        'price': ['mean', 'std']
    })
    city_variation.columns = ['mean_price', 'std_price']
    city_variation['cv'] = city_variation['std_price'] / city_variation['mean_price']
    city_variation = city_variation.sort_values('cv', ascending=False)
    
    plt.figure(figsize=(10, 6))
    city_variation['cv'].plot(kind='bar', color=colors[4])
    plt.title('Coefficient de variation des prix par ville', fontsize=16)
    plt.xlabel('Ville', fontsize=14)
    plt.ylabel('Coefficient de variation', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fig_count:02d}_variation_prix_par_ville.png", dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    
    # Sauvegarder l'analyse de variation
    city_variation.to_csv(f"{output_dir}/{fig_count:02d}_variation_prix_par_ville.csv")
    fig_count += 1
    
    # 11. Analyse des quartiers premium
    print("\n=== Analyse des quartiers premium ===")
    
    # Identifier les quartiers premium (prix > 75ème percentile)
    price_75th = df['price'].quantile(0.75)
    premium_threshold = price_75th * 1.5  # 50% au-dessus du 75ème percentile
    
    # Quartiers avec proportion élevée de logements premium
    neighbourhood_premium = df.groupby(['city', 'neighbourhood']).agg({
        'price': ['count', 'mean', 'median'],
        'id': 'count'
    })
    neighbourhood_premium.columns = ['count', 'mean_price', 'median_price', 'total_count']
    neighbourhood_premium['premium_ratio'] = (
        df[df['price'] > premium_threshold]
        .groupby(['city', 'neighbourhood'])
        .size() / neighbourhood_premium['count']
    ).fillna(0)
    
    # Filtrer les quartiers avec au moins 10 propriétés
    neighbourhood_premium = neighbourhood_premium[neighbourhood_premium['count'] >= 10]
    
    # Top 10 quartiers premium
    top_premium = neighbourhood_premium.sort_values('premium_ratio', ascending=False).head(10)
    
    plt.figure(figsize=(14, 8))
    top_premium['premium_ratio'].plot(kind='bar', color=colors[0])
    plt.title('Top 10 des quartiers premium (% de logements haut de gamme)', fontsize=16)
    plt.xlabel('Quartier (Ville)', fontsize=14)
    plt.ylabel('Proportion de logements premium', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fig_count:02d}_quartiers_premium.png", dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    
    # Sauvegarder l'analyse des quartiers premium
    top_premium.to_csv(f"{output_dir}/{fig_count:02d}_quartiers_premium.csv")
    fig_count += 1
    
    # 12. Conclusions et observations
    print("\n=== Principales observations sur l'impact géographique ===")
    
    # Impact des villes
    city_impact = df.groupby('city')['log_price'].mean().sort_values(ascending=False)
    print("\nImpact des villes (log_price moyen):")
    for city, price in city_impact.items():
        print(f"- {city}: {price:.4f}")
    
    # Villes avec le plus de variabilité
    city_variability = df.groupby('city')['log_price'].std().sort_values(ascending=False)
    print("\nVariabilité des prix par ville (écart-type log_price):")
    for city, std in city_variability.items():
        print(f"- {city}: {std:.4f}")
    
    # Corrélations géographiques (si disponibles)
    if 'latitude' in df.columns and 'longitude' in df.columns:
        lat_corr = df['latitude'].corr(df['log_price'])
        lon_corr = df['longitude'].corr(df['log_price'])
        print(f"\nCorreélation latitude-prix: {lat_corr:.4f}")
        print(f"Corrélation longitude-prix: {lon_corr:.4f}")
    
    # Sauvegarder un résumé des observations
    with open(f"{output_dir}/{fig_count:02d}_resume_observations_geo.txt", 'w', encoding='utf-8') as f:
        f.write("=== RÉSUMÉ DES OBSERVATIONS SUR L'IMPACT GÉOGRAPHIQUE ===\n\n")
        
        f.write("Impact des villes (log_price moyen):\n")
        for city, price in city_impact.items():
            f.write(f"- {city}: {price:.4f}\n")
            
        f.write("\nVariabilité des prix par ville (écart-type log_price):\n")
        for city, std in city_variability.items():
            f.write(f"- {city}: {std:.4f}\n")
            
        if 'latitude' in df.columns and 'longitude' in df.columns:
            f.write(f"\nCorreélation latitude-prix: {lat_corr:.4f}\n")
            f.write(f"Corrélation longitude-prix: {lon_corr:.4f}\n")
            
        f.write(f"\nNombre total de quartiers analysés: {df['neighbourhood'].nunique()}\n")
        f.write(f"Nombre de villes: {df['city'].nunique()}\n")
    
    print(f"\nAnalyse terminée. {fig_count} visualisations enregistrées dans {output_dir}")
    return df

# Exécuter l'analyse
if __name__ == "__main__":
    file_path = "Data/Clean/train_partial_clean_standart.csv"
    output_dir = "Data/Visual/plot/Multiple/localisation"
    df_analyzed = analyze_location_impact(file_path, output_dir)