import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# Configuration pour de meilleures visualisations
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Couleurs Airbnb
colors = ["#FF5A5F", "#00A699", "#FC642D", "#484848", "#767676"]
sns.set_palette(sns.color_palette(colors))

def analyze_physical_characteristics(file_path, output_dir="Data/Visual/plot/Multiple/cara_physique"):
    """
    Analyse approfondie des caractéristiques physiques des propriétés
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
    
    # Liste des caractéristiques physiques
    physical_features = ['property_type', 'room_type', 'accommodates', 
                        'bedrooms', 'beds', 'bed_type', 'bathrooms']
    
    # 1. Statistiques descriptives
    print("\n=== Statistiques descriptives des caractéristiques physiques ===")
    numeric_features = [f for f in physical_features if df[f].dtype in ['int64', 'float64']]
    
    stats = df[numeric_features].describe()
    print(stats)
    
    # Sauvegarder les statistiques dans un CSV
    stats.to_csv(f"{output_dir}/01_stats_descriptives.csv")
    
    # 2. Distribution des types de propriétés
    print("\n=== Distribution des types de propriétés ===")
    property_counts = df['property_type'].value_counts()
    print(property_counts)
    
    plt.figure(figsize=(14, 6))
    property_counts.plot(kind='bar', color=colors[1])
    plt.title('Distribution des types de propriétés', fontsize=16)
    plt.xlabel('Type de propriété', fontsize=14)
    plt.ylabel('Nombre', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fig_count:02d}_distribution_types_proprietes.png", dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    
    # 3. Relation entre le type de propriété et le prix
    plt.figure(figsize=(14, 6))
    property_median = df.groupby('property_type')['price'].median().sort_values(ascending=False)
    property_median.plot(kind='bar', color=colors[2])
    plt.title('Prix médian par type de propriété', fontsize=16)
    plt.xlabel('Type de propriété', fontsize=14)
    plt.ylabel('Prix médian ($)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fig_count:02d}_prix_median_par_type_propriete.png", dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    
    # 4. Distribution des types de chambre
    print("\n=== Distribution des types de chambre ===")
    room_counts = df['room_type'].value_counts()
    print(room_counts)
    
    plt.figure(figsize=(10, 6))
    room_counts.plot(kind='bar', color=colors[3])
    plt.title('Distribution des types de chambre', fontsize=16)
    plt.xlabel('Type de chambre', fontsize=14)
    plt.ylabel('Nombre', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fig_count:02d}_distribution_types_chambre.png", dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    
    # 5. Relation entre le type de chambre et le prix
    plt.figure(figsize=(10, 6))
    room_median = df.groupby('room_type')['price'].median().sort_values(ascending=False)
    room_median.plot(kind='bar', color=colors[4])
    plt.title('Prix médian par type de chambre', fontsize=16)
    plt.xlabel('Type de chambre', fontsize=14)
    plt.ylabel('Prix médian ($)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fig_count:02d}_prix_median_par_type_chambre.png", dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    
    # 6. Relations entre les caractéristiques numériques et le prix
    numeric_features = ['accommodates', 'bedrooms', 'beds', 'bathrooms']
    
    # Graphiques de régression
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(numeric_features):
        sns.regplot(x=feature, y='log_price', data=df, 
                   scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':colors[0]}, ax=axes[i])
        axes[i].set_title(f'Relation entre {feature} et log_price', fontsize=14)
        axes[i].set_xlabel(feature, fontsize=12)
        axes[i].set_ylabel('log_price', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fig_count:02d}_relation_caracteristiques_numeriques_prix.png", dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    
    # 7. Distribution des types de lit
    print("\n=== Distribution des types de lit ===")
    bed_counts = df['bed_type'].value_counts()
    print(bed_counts)
    
    plt.figure(figsize=(10, 6))
    bed_counts.plot(kind='bar', color=colors[0])
    plt.title('Distribution des types de lit', fontsize=16)
    plt.xlabel('Type de lit', fontsize=14)
    plt.ylabel('Nombre', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fig_count:02d}_distribution_types_lit.png", dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    
    # 8. Relation entre le type de lit et le prix
    plt.figure(figsize=(10, 6))
    bed_median = df.groupby('bed_type')['price'].median().sort_values(ascending=False)
    bed_median.plot(kind='bar', color=colors[1])
    plt.title('Prix médian par type de lit', fontsize=16)
    plt.xlabel('Type de lit', fontsize=14)
    plt.ylabel('Prix médian ($)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fig_count:02d}_prix_median_par_type_lit.png", dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    
    # 9. Analyse plus détaillée: prix par personne et par chambre
    df['price_per_person'] = df['price'] / df['accommodates']
    df['price_per_bedroom'] = df['price'] / df['bedrooms'].replace(0, 1)  # Éviter division par zéro
    
    # Moyenne de prix par personne par type de chambre
    plt.figure(figsize=(10, 6))
    person_price_by_room = df.groupby('room_type')['price_per_person'].mean().sort_values(ascending=False)
    person_price_by_room.plot(kind='bar', color=colors[2])
    plt.title('Prix moyen par personne selon le type de chambre', fontsize=14)
    plt.xlabel('Type de chambre', fontsize=12)
    plt.ylabel('Prix moyen par personne ($)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fig_count:02d}_prix_moyen_par_personne_type_chambre.png", dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    
    # Moyenne de prix par chambre par type de propriété
    plt.figure(figsize=(14, 6))
    bedroom_price_by_property = df.groupby('property_type')['price_per_bedroom'].mean().sort_values(ascending=False)
    bedroom_price_by_property.plot(kind='bar', color=colors[3])
    plt.title('Prix moyen par chambre selon le type de propriété', fontsize=14)
    plt.xlabel('Type de propriété', fontsize=12)
    plt.ylabel('Prix moyen par chambre ($)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fig_count:02d}_prix_moyen_par_chambre_type_propriete.png", dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    
    # 10. Analyse approfondie des amenities
    if 'amenities' in df.columns:
        print("\n=== Analyse approfondie des aménités ===")
        
        # Fonction pour extraire les amenities
        def extract_amenities(amenities_str):
            try:
                if amenities_str.startswith('{') and amenities_str.endswith('}'):
                    amenities_str = amenities_str[1:-1]
                return re.findall(r'"([^"]*)"', amenities_str)
            except:
                return []
        
        # Comptage du nombre d'aménités
        df['amenities_count'] = df['amenities'].apply(lambda x: len(extract_amenities(x)))
        
        # Relation entre le nombre d'aménités et le prix
        plt.figure(figsize=(10, 6))
        sns.regplot(x='amenities_count', y='log_price', data=df, 
                   scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':colors[2]})
        plt.title('Relation entre le nombre d\'aménités et log_price', fontsize=16)
        plt.xlabel('Nombre d\'aménités', fontsize=14)
        plt.ylabel('log_price', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fig_count:02d}_relation_nombre_amenites_prix.png", dpi=300, bbox_inches='tight')
        plt.close()
        fig_count += 1
        
        # Extraction de toutes les aménités pour trouver les plus courantes
        all_amenities = []
        for amenities_str in df['amenities']:
            all_amenities.extend(extract_amenities(amenities_str))
        
        # Compter les occurrences de chaque aménité
        amenities_count = pd.Series(all_amenities).value_counts()
        
        # Afficher les 20 aménités les plus courantes
        plt.figure(figsize=(16, 8))
        amenities_count.head(20).plot(kind='bar', color=colors[0])
        plt.title('Les 20 aménités les plus courantes', fontsize=16)
        plt.xlabel('Aménité', fontsize=14)
        plt.ylabel('Nombre d\'occurrences', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fig_count:02d}_20_amenites_plus_courantes.png", dpi=300, bbox_inches='tight')
        plt.close()
        fig_count += 1
        
        # Analyser l'impact de chaque aménité sur le prix
        top_amenities = amenities_count.head(15).index.tolist()
        amenity_impact = {}
        
        for amenity in top_amenities:
            # Créer une colonne binaire pour cette aménité
            df[f'has_{amenity}'] = df['amenities'].apply(
                lambda x: 1 if amenity in x else 0
            )
            
            # Calculer les prix avec et sans cette aménité
            with_amenity = df[df[f'has_{amenity}'] == 1]['price'].mean()
            without_amenity = df[df[f'has_{amenity}'] == 0]['price'].mean()
            
            # Calculer la différence en pourcentage
            if without_amenity > 0:
                diff_pct = (with_amenity - without_amenity) / without_amenity * 100
            else:
                diff_pct = 0
                
            # Stocker les résultats
            amenity_impact[amenity] = {
                'with_amenity': with_amenity,
                'without_amenity': without_amenity,
                'diff_pct': diff_pct,
                'count': amenities_count[amenity]
            }
        
        # Convertir en DataFrame pour visualisation
        impact_df = pd.DataFrame.from_dict(amenity_impact, orient='index')
        impact_df = impact_df.sort_values('diff_pct', ascending=False)
        
        # Sauvegarder les données d'impact dans un CSV
        impact_df.to_csv(f"{output_dir}/{fig_count:02d}_impact_amenities_sur_prix.csv")
        fig_count += 1
        
        # Graphique de l'impact des aménités sur le prix
        plt.figure(figsize=(16, 8))
        impact_df['diff_pct'].plot(kind='bar', color=colors[1])
        plt.title('Impact des aménités sur le prix (% de différence)', fontsize=16)
        plt.xlabel('Aménité', fontsize=14)
        plt.ylabel('Différence de prix en %', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fig_count:02d}_impact_amenites_sur_prix_pct.png", dpi=300, bbox_inches='tight')
        plt.close()
        fig_count += 1
        
        # Graphique comparatif des prix avec/sans amenities spécifiques
        top_impact_amenities = impact_df.sort_values('diff_pct', ascending=False).head(8).index.tolist()
        
        comparison_data = []
        for amenity in top_impact_amenities:
            comparison_data.append({
                'amenity': amenity,
                'type': 'Avec',
                'price': impact_df.loc[amenity, 'with_amenity']
            })
            comparison_data.append({
                'amenity': amenity,
                'type': 'Sans',
                'price': impact_df.loc[amenity, 'without_amenity']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        plt.figure(figsize=(16, 8))
        sns.barplot(x='amenity', y='price', hue='type', data=comparison_df, palette=[colors[2], colors[4]])
        plt.title('Comparaison des prix moyens avec/sans aménités spécifiques', fontsize=16)
        plt.xlabel('Aménité', fontsize=14)
        plt.ylabel('Prix moyen ($)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fig_count:02d}_comparaison_prix_avec_sans_amenites.png", dpi=300, bbox_inches='tight')
        plt.close()
        fig_count += 1
        
        # Analyse des combinaisons d'aménités premium
        print("\n=== Analyse des combinaisons d'aménités premium ===")
        
        # Identifier les aménités "premium" (celles avec le plus grand impact positif sur le prix)
        premium_amenities = impact_df.sort_values('diff_pct', ascending=False).head(5).index.tolist()
        print(f"Les 5 aménités premium identifiées: {premium_amenities}")
        
        # Calculer un score premium basé sur ces aménités
        df['premium_score'] = sum(df[f'has_{amenity}'] for amenity in premium_amenities)
        
        # Relation entre le score premium et le prix
        plt.figure(figsize=(10, 6))
        premium_price = df.groupby('premium_score')['price'].mean()
        premium_price.plot(kind='bar', color=colors[3])
        plt.title('Prix moyen par nombre d\'aménités premium', fontsize=16)
        plt.xlabel('Nombre d\'aménités premium', fontsize=14)
        plt.ylabel('Prix moyen ($)', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fig_count:02d}_prix_moyen_par_nombre_amenites_premium.png", dpi=300, bbox_inches='tight')
        plt.close()
        fig_count += 1
    
    # 11. Matrice de corrélation (à la fin comme demandé)
    print("\n=== Corrélation entre les caractéristiques numériques et le prix ===")
    extended_numeric = numeric_features + ['log_price', 'price']
    if 'amenities_count' in df.columns:
        extended_numeric.append('amenities_count')
    if 'premium_score' in df.columns:
        extended_numeric.append('premium_score')
    if 'price_per_person' in df.columns:
        extended_numeric.append('price_per_person')
    if 'price_per_bedroom' in df.columns:
        extended_numeric.append('price_per_bedroom')
    
    plt.figure(figsize=(12, 10))
    corr_matrix = df[extended_numeric].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, fmt='.2f',
               square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Matrice de corrélation des caractéristiques', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fig_count:02d}_matrice_correlation.png", dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    
    # Sauvegarder la matrice de corrélation dans un CSV
    corr_matrix.to_csv(f"{output_dir}/{fig_count:02d}_matrice_correlation.csv")
    fig_count += 1
    
    # 12. Conclusions et observations
    print("\n=== Principales observations sur les caractéristiques physiques ===")
    
    # Corrélations numériques
    corr_with_price = corr_matrix['log_price'].drop('log_price').sort_values(ascending=False)
    print("\nCorrélation avec log_price (caractéristiques numériques):")
    for feature, corr in corr_with_price.items():
        print(f"- {feature}: {corr:.4f}")
    
    # Impact des types de propriété
    prop_impact = df.groupby('property_type')['log_price'].mean().sort_values(ascending=False)
    print("\nImpact du type de propriété (log_price moyen):")
    for prop, price in prop_impact.head(3).items():
        print(f"- {prop}: {price:.4f}")
    
    # Impact des types de chambre
    room_impact = df.groupby('room_type')['log_price'].mean().sort_values(ascending=False)
    print("\nImpact du type de chambre (log_price moyen):")
    for room, price in room_impact.items():
        print(f"- {room}: {price:.4f}")
    
    # Impact des aménités (si analysées)
    if 'amenities' in df.columns and 'impact_df' in locals():
        print("\nAménités avec le plus grand impact sur le prix:")
        for amenity, impact in impact_df['diff_pct'].head(5).items():
            print(f"- {amenity}: +{impact:.2f}%")
    
    # Sauvegarder un résumé des observations dans un fichier texte
    with open(f"{output_dir}/{fig_count:02d}_resume_observations.txt", 'w') as f:
        f.write("=== RÉSUMÉ DES OBSERVATIONS SUR LES CARACTÉRISTIQUES PHYSIQUES ===\n\n")
        
        f.write("Corrélation avec log_price (caractéristiques numériques):\n")
        for feature, corr in corr_with_price.items():
            f.write(f"- {feature}: {corr:.4f}\n")
        
        f.write("\nImpact du type de propriété (log_price moyen):\n")
        for prop, price in prop_impact.head(3).items():
            f.write(f"- {prop}: {price:.4f}\n")
        
        f.write("\nImpact du type de chambre (log_price moyen):\n")
        for room, price in room_impact.items():
            f.write(f"- {room}: {price:.4f}\n")
        
        if 'amenities' in df.columns and 'impact_df' in locals():
            f.write("\nAménités avec le plus grand impact sur le prix:\n")
            for amenity, impact in impact_df['diff_pct'].head(5).items():
                f.write(f"- {amenity}: +{impact:.2f}%\n")
    
    print(f"\nAnalyse terminée. {fig_count} visualisations enregistrées dans {output_dir}")
    return df

# Exécuter l'analyse
file_path = "Data/Clean/train_partial_clean_standart.csv"
output_dir = "Data/Visual/plot/Multiple/cara_physique"
df_analyzed = analyze_physical_characteristics(file_path, output_dir)