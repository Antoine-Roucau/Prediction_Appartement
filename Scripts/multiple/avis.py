import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from scipy import stats

# Configuration pour de meilleures visualisations
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Couleurs Airbnb
colors = ["#FF5A5F", "#00A699", "#FC642D", "#484848", "#767676"]
sns.set_palette(sns.color_palette(colors))

def analyze_hosts_reviews_impact(file_path, output_dir="Data/Visual/plot/Multiple/hotes_avis"):
    """
    Analyse approfondie de l'impact des hôtes et des avis sur les prix Airbnb
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
    
    # Variables liées aux hôtes et avis
    host_review_features = ['host_since', 'host_identity_verified', 
                           'number_of_reviews', 'review_scores_rating']
    
    # 1. Préparation des données sur l'expérience des hôtes
    print("\n=== Préparation des données d'expérience des hôtes ===")
    
    if 'host_since' in df.columns:
        # Convertir host_since en datetime
        df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
        
        # Calculer l'expérience en années (depuis aujourd'hui)
        current_date = pd.to_datetime('2024-01-01')  # Date de référence
        df['host_experience_years'] = (current_date - df['host_since']).dt.days / 365.25
        
        # Nettoyer les valeurs aberrantes (expérience négative ou trop élevée)
        df['host_experience_years'] = df['host_experience_years'].clip(0, 20)
        
        print(f"Expérience des hôtes calculée pour {df['host_experience_years'].notna().sum()} hôtes")
    
    # Statistiques descriptives générales
    stats_summary = []
    
    # 2. Analyse de l'expérience des hôtes
    if 'host_experience_years' in df.columns:
        print("\n=== Analyse de l'expérience des hôtes ===")
        
        # Statistiques descriptives
        exp_stats = df['host_experience_years'].describe()
        print("Statistiques sur l'expérience des hôtes (années):")
        print(exp_stats)
        
        # Sauvegarder les statistiques
        exp_stats.to_csv(f"{output_dir}/01_stats_experience_hotes.csv")
        
        # Distribution de l'expérience des hôtes
        plt.figure(figsize=(12, 6))
        sns.histplot(df['host_experience_years'].dropna(), bins=30, kde=True, color=colors[0])
        plt.title('Distribution de l\'expérience des hôtes', fontsize=16)
        plt.xlabel('Années d\'expérience', fontsize=14)
        plt.ylabel('Fréquence', fontsize=14)
        plt.axvline(df['host_experience_years'].mean(), color=colors[2], linestyle='--', 
                   label=f'Moyenne: {df["host_experience_years"].mean():.1f} ans')
        plt.axvline(df['host_experience_years'].median(), color=colors[1], linestyle='-', 
                   label=f'Médiane: {df["host_experience_years"].median():.1f} ans')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fig_count:02d}_distribution_experience_hotes.png", dpi=300, bbox_inches='tight')
        plt.close()
        fig_count += 1
        
        # Relation expérience-prix
        plt.figure(figsize=(14, 8))
        sns.regplot(x='host_experience_years', y='log_price', data=df.dropna(), 
                   scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':colors[0]})
        plt.title('Relation entre l\'expérience de l\'hôte et le prix', fontsize=16)
        plt.xlabel('Années d\'expérience', fontsize=14)
        plt.ylabel('log_price', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fig_count:02d}_relation_experience_prix.png", dpi=300, bbox_inches='tight')
        plt.close()
        fig_count += 1
        
        # Catégorisation de l'expérience
        df['experience_category'] = pd.cut(
            df['host_experience_years'],
            bins=[0, 1, 3, 6, float('inf')],
            labels=['Nouveau (0-1 an)', 'Junior (1-3 ans)', 'Expérimenté (3-6 ans)', 'Vétéran (6+ ans)']
        )
        
        # Prix par catégorie d'expérience
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='experience_category', y='price', 
                   data=df.dropna(subset=['experience_category', 'price']))
        plt.title('Prix par catégorie d\'expérience d\'hôte', fontsize=16)
        plt.xlabel('Catégorie d\'expérience', fontsize=14)
        plt.ylabel('Prix ($)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fig_count:02d}_prix_par_categorie_experience.png", dpi=300, bbox_inches='tight')
        plt.close()
        fig_count += 1
        
        # Prix médian par catégorie d'expérience
        plt.figure(figsize=(10, 6))
        exp_median = df.groupby('experience_category')['price'].median().dropna()
        exp_median.plot(kind='bar', color=colors[1])
        plt.title('Prix médian par catégorie d\'expérience d\'hôte', fontsize=16)
        plt.xlabel('Catégorie d\'expérience', fontsize=14)
        plt.ylabel('Prix médian ($)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fig_count:02d}_prix_median_par_experience.png", dpi=300, bbox_inches='tight')
        plt.close()
        fig_count += 1
    
    # 3. Analyse de la vérification d'identité
    if 'host_identity_verified' in df.columns:
        print("\n=== Analyse de la vérification d'identité ===")
        
        # Distribution de la vérification
        verification_counts = df['host_identity_verified'].value_counts()
        verification_pcts = df['host_identity_verified'].value_counts(normalize=True) * 100
        
        print("Distribution des hôtes par statut de vérification:")
        for status, count in verification_counts.items():
            print(f"- {status}: {count} hôtes ({verification_pcts[status]:.1f}%)")
        
        # Graphique de distribution
        plt.figure(figsize=(8, 6))
        verification_counts.plot(kind='bar', color=colors[2])
        plt.title('Distribution des hôtes par statut de vérification d\'identité', fontsize=16)
        plt.xlabel('Identité vérifiée', fontsize=14)
        plt.ylabel('Nombre d\'hôtes', fontsize=14)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fig_count:02d}_distribution_verification_identite.png", dpi=300, bbox_inches='tight')
        plt.close()
        fig_count += 1
        
        # Comparaison des prix selon la vérification
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='host_identity_verified', y='price', 
                   data=df[df['price'] <= df['price'].quantile(0.95)])
        plt.title('Distribution des prix selon la vérification d\'identité', fontsize=16)
        plt.xlabel('Identité vérifiée', fontsize=14)
        plt.ylabel('Prix ($)', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fig_count:02d}_prix_par_verification_identite.png", dpi=300, bbox_inches='tight')
        plt.close()
        fig_count += 1
        
        # Prix médian par statut de vérification
        verification_impact = df.groupby('host_identity_verified').agg({
            'price': ['count', 'mean', 'median'],
            'log_price': ['mean', 'median']
        }).round(2)
        
        plt.figure(figsize=(8, 6))
        medians = df.groupby('host_identity_verified')['price'].median()
        medians.plot(kind='bar', color=colors[3])
        plt.title('Prix médian selon le statut de vérification d\'identité', fontsize=16)
        plt.xlabel('Identité vérifiée', fontsize=14)
        plt.ylabel('Prix médian ($)', fontsize=14)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fig_count:02d}_prix_median_verification.png", dpi=300, bbox_inches='tight')
        plt.close()
        fig_count += 1
        
        # Sauvegarder l'analyse de vérification
        verification_impact.to_csv(f"{output_dir}/{fig_count:02d}_impact_verification.csv")
        fig_count += 1
    
    # 4. Analyse du nombre d'avis
    print("\n=== Analyse du nombre d'avis ===")
    
    # Statistiques descriptives des avis
    review_stats = df['number_of_reviews'].describe()
    print("Statistiques sur le nombre d'avis:")
    print(review_stats)
    
    # Distribution du nombre d'avis
    plt.figure(figsize=(14, 8))
    # Utiliser une échelle log pour mieux voir la distribution
    plt.subplot(1, 2, 1)
    sns.histplot(df['number_of_reviews'], bins=50, kde=True, color=colors[0])
    plt.title('Distribution du nombre d\'avis', fontsize=14)
    plt.xlabel('Nombre d\'avis', fontsize=12)
    plt.ylabel('Fréquence', fontsize=12)
    
    plt.subplot(1, 2, 2)
    sns.histplot(df[df['number_of_reviews'] > 0]['number_of_reviews'], 
                bins=50, kde=True, log_scale=True, color=colors[1])
    plt.title('Distribution du nombre d\'avis (échelle log)', fontsize=14)
    plt.xlabel('Nombre d\'avis', fontsize=12)
    plt.ylabel('Fréquence', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fig_count:02d}_distribution_nombre_avis.png", dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    
    # Relation nombre d'avis - prix
    plt.figure(figsize=(14, 8))
    sns.regplot(x='number_of_reviews', y='log_price', data=df,
               scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':colors[2]})
    plt.title('Relation entre le nombre d\'avis et le prix', fontsize=16)
    plt.xlabel('Nombre d\'avis', fontsize=14)
    plt.ylabel('log_price', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fig_count:02d}_relation_nombre_avis_prix.png", dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    
    # Catégorisation du nombre d'avis
    df['reviews_category'] = pd.cut(
        df['number_of_reviews'],
        bins=[0, 1, 10, 50, float('inf')],
        labels=['Aucun avis (0)', 'Peu d\'avis (1-10)', 'Modéré (10-50)', 'Beaucoup (50+)'],
        include_lowest=True
    )
    
    # Prix par catégorie d'avis
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='reviews_category', y='price', 
               data=df[df['price'] <= df['price'].quantile(0.95)])
    plt.title('Prix par catégorie de nombre d\'avis', fontsize=16)
    plt.xlabel('Catégorie d\'avis', fontsize=14)
    plt.ylabel('Prix ($)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fig_count:02d}_prix_par_categorie_avis.png", dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    
    # 5. Analyse des scores d'évaluation
    if 'review_scores_rating' in df.columns:
        print("\n=== Analyse des scores d'évaluation ===")
        
        # Nettoyer les données de rating
        df_rating = df.dropna(subset=['review_scores_rating'])
        
        if len(df_rating) > 0:
            # Statistiques des scores
            rating_stats = df_rating['review_scores_rating'].describe()
            print("Statistiques sur les scores d'évaluation:")
            print(rating_stats)
            
            # Distribution des scores
            plt.figure(figsize=(12, 6))
            sns.histplot(df_rating['review_scores_rating'], bins=30, kde=True, color=colors[3])
            plt.title('Distribution des scores d\'évaluation', fontsize=16)
            plt.xlabel('Score d\'évaluation', fontsize=14)
            plt.ylabel('Fréquence', fontsize=14)
            plt.axvline(df_rating['review_scores_rating'].mean(), color=colors[2], linestyle='--', 
                       label=f'Moyenne: {df_rating["review_scores_rating"].mean():.1f}')
            plt.axvline(df_rating['review_scores_rating'].median(), color=colors[1], linestyle='-', 
                       label=f'Médiane: {df_rating["review_scores_rating"].median():.1f}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{fig_count:02d}_distribution_scores_evaluation.png", dpi=300, bbox_inches='tight')
            plt.close()
            fig_count += 1
            
            # Relation score-prix
            plt.figure(figsize=(14, 8))
            sns.regplot(x='review_scores_rating', y='log_price', data=df_rating,
                       scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':colors[4]})
            plt.title('Relation entre le score d\'évaluation et le prix', fontsize=16)
            plt.xlabel('Score d\'évaluation', fontsize=14)
            plt.ylabel('log_price', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{fig_count:02d}_relation_score_prix.png", dpi=300, bbox_inches='tight')
            plt.close()
            fig_count += 1
            
            # Catégorisation des scores
            df['rating_category'] = pd.cut(
                df['review_scores_rating'],
                bins=[0, 80, 90, 95, 100],
                labels=['Faible (0-80)', 'Moyen (80-90)', 'Bon (90-95)', 'Excellent (95-100)'],
                include_lowest=True
            )
            
            # Prix par catégorie de score
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='rating_category', y='price', 
                       data=df.dropna(subset=['rating_category']))
            plt.title('Prix par catégorie de score d\'évaluation', fontsize=16)
            plt.xlabel('Catégorie de score', fontsize=14)
            plt.ylabel('Prix ($)', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{fig_count:02d}_prix_par_categorie_score.png", dpi=300, bbox_inches='tight')
            plt.close()
            fig_count += 1
            
            # Prix médian par catégorie de score
            plt.figure(figsize=(10, 6))
            score_median = df.groupby('rating_category')['price'].median().dropna()
            score_median.plot(kind='bar', color=colors[0])
            plt.title('Prix médian par catégorie de score d\'évaluation', fontsize=16)
            plt.xlabel('Catégorie de score', fontsize=14)
            plt.ylabel('Prix médian ($)', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{fig_count:02d}_prix_median_par_score.png", dpi=300, bbox_inches='tight')
            plt.close()
            fig_count += 1
    
    # 6. Analyse combinée : indice de réputation
    print("\n=== Création d'un indice de réputation combiné ===")
    
    # Normaliser les différents facteurs de réputation
    reputation_factors = {}
    
    # Facteur expérience (si disponible)
    if 'host_experience_years' in df.columns:
        # Normaliser l'expérience sur une échelle 0-1
        max_exp = df['host_experience_years'].quantile(0.95)  # éviter les outliers
        df['exp_score'] = (df['host_experience_years'] / max_exp).clip(0, 1)
        reputation_factors['exp_score'] = 0.2
    
    # Facteur vérification
    if 'host_identity_verified' in df.columns:
        df['verified_score'] = (df['host_identity_verified'] == 't').astype(int)
        reputation_factors['verified_score'] = 0.1
    
    # Facteur nombre d'avis (normaliser avec log pour éviter les outliers)
    df['reviews_score'] = np.log1p(df['number_of_reviews']) / np.log1p(df['number_of_reviews'].quantile(0.95))
    df['reviews_score'] = df['reviews_score'].clip(0, 1)
    reputation_factors['reviews_score'] = 0.4
    
    # Facteur score d'évaluation
    if 'review_scores_rating' in df.columns:
        df['rating_score'] = (df['review_scores_rating'] - 60) / 40  # normaliser 60-100 vers 0-1
        df['rating_score'] = df['rating_score'].clip(0, 1).fillna(0.5)  # moyenne pour les manquants
        reputation_factors['rating_score'] = 0.3
    
    # Calculer l'indice de réputation combiné
    df['reputation_index'] = 0
    for factor, weight in reputation_factors.items():
        df['reputation_index'] += df[factor].fillna(0) * weight
    
    # Normaliser l'indice final sur 0-10
    df['reputation_index'] = df['reputation_index'] * 10
    
    # Distribution de l'indice de réputation
    plt.figure(figsize=(12, 6))
    sns.histplot(df['reputation_index'], bins=30, kde=True, color=colors[2])
    plt.title('Distribution de l\'indice de réputation combiné', fontsize=16)
    plt.xlabel('Indice de réputation (0-10)', fontsize=14)
    plt.ylabel('Fréquence', fontsize=14)
    plt.axvline(df['reputation_index'].mean(), color=colors[0], linestyle='--', 
               label=f'Moyenne: {df["reputation_index"].mean():.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fig_count:02d}_distribution_indice_reputation.png", dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    
    # Relation indice de réputation - prix
    plt.figure(figsize=(14, 8))
    sns.regplot(x='reputation_index', y='log_price', data=df,
               scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':colors[1]})
    plt.title('Relation entre l\'indice de réputation et le prix', fontsize=16)
    plt.xlabel('Indice de réputation (0-10)', fontsize=14)
    plt.ylabel('log_price', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fig_count:02d}_relation_reputation_prix.png", dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    
    # Catégorisation de la réputation
    df['reputation_category'] = pd.cut(
        df['reputation_index'],
        bins=[0, 3, 6, 8, 10],
        labels=['Faible (0-3)', 'Modérée (3-6)', 'Bonne (6-8)', 'Excellente (8-10)']
    )
    
    # Prix par catégorie de réputation
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='reputation_category', y='price', 
               data=df[df['price'] <= df['price'].quantile(0.95)])
    plt.title('Prix par catégorie de réputation', fontsize=16)
    plt.xlabel('Catégorie de réputation', fontsize=14)
    plt.ylabel('Prix ($)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fig_count:02d}_prix_par_categorie_reputation.png", dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    
    # 7. Matrice de corrélation des facteurs de réputation
    print("\n=== Matrice de corrélation des facteurs de réputation ===")
    
    reputation_columns = ['log_price', 'price']
    if 'host_experience_years' in df.columns:
        reputation_columns.append('host_experience_years')
    reputation_columns.extend(['number_of_reviews', 'reputation_index'])
    if 'review_scores_rating' in df.columns:
        reputation_columns.append('review_scores_rating')
    
    corr_matrix = df[reputation_columns].corr()
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
               square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Matrice de corrélation des facteurs de réputation', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fig_count:02d}_correlation_reputation.png", dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    
    # Sauvegarder la matrice de corrélation
    corr_matrix.to_csv(f"{output_dir}/{fig_count:02d}_correlation_reputation.csv")
    fig_count += 1
    
    # 8. Analyse comparative des impacts
    print("\n=== Analyse comparative des impacts ===")
    
    # Calcul des impacts moyens
    impact_analysis = {}
    
    # Impact de la vérification
    if 'host_identity_verified' in df.columns:
        verified_price = df[df['host_identity_verified'] == 't']['price'].mean()
        not_verified_price = df[df['host_identity_verified'] == 'f']['price'].mean()
        impact_analysis['Vérification identité'] = (verified_price - not_verified_price) / not_verified_price * 100
    
    # Impact de l'expérience (vétéran vs nouveau)
    if 'experience_category' in df.columns:
        veteran_price = df[df['experience_category'] == 'Vétéran (6+ ans)']['price'].mean()
        newbie_price = df[df['experience_category'] == 'Nouveau (0-1 an)']['price'].mean()
        if not pd.isna(veteran_price) and not pd.isna(newbie_price):
            impact_analysis['Expérience (vétéran vs nouveau)'] = (veteran_price - newbie_price) / newbie_price * 100
    
    # Impact des avis (beaucoup vs aucun)
    if 'reviews_category' in df.columns:
        many_reviews_price = df[df['reviews_category'] == 'Beaucoup (50+)']['price'].mean()
        no_reviews_price = df[df['reviews_category'] == 'Aucun avis (0)']['price'].mean()
        if not pd.isna(many_reviews_price) and not pd.isna(no_reviews_price):
            impact_analysis['Nombre d\'avis (50+ vs 0)'] = (many_reviews_price - no_reviews_price) / no_reviews_price * 100
    
    # Impact du score (excellent vs faible)
    if 'rating_category' in df.columns:
        excellent_price = df[df['rating_category'] == 'Excellent (95-100)']['price'].mean()
        poor_price = df[df['rating_category'] == 'Faible (0-80)']['price'].mean()
        if not pd.isna(excellent_price) and not pd.isna(poor_price):
            impact_analysis['Score (excellent vs faible)'] = (excellent_price - poor_price) / poor_price * 100
    
    # Impact de la réputation (excellente vs faible)
    excellent_rep_price = df[df['reputation_category'] == 'Excellente (8-10)']['price'].mean()
    poor_rep_price = df[df['reputation_category'] == 'Faible (0-3)']['price'].mean()
    if not pd.isna(excellent_rep_price) and not pd.isna(poor_rep_price):
        impact_analysis['Réputation globale (excellente vs faible)'] = (excellent_rep_price - poor_rep_price) / poor_rep_price * 100
    
    # Graphique des impacts
    if impact_analysis:
        impact_df = pd.DataFrame(list(impact_analysis.items()), columns=['Facteur', 'Impact (%)'])
        impact_df = impact_df.sort_values('Impact (%)', ascending=True)
        
        plt.figure(figsize=(14, 8))
        plt.barh(impact_df['Facteur'], impact_df['Impact (%)'], color=colors[3])
        plt.title('Impact comparatif des facteurs de réputation sur les prix', fontsize=16)
        plt.xlabel('Impact sur le prix (%)', fontsize=14)
        plt.ylabel('Facteur de réputation', fontsize=14)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fig_count:02d}_impact_comparatif_reputation.png", dpi=300, bbox_inches='tight')
        plt.close()
        fig_count += 1
        
        # Sauvegarder l'analyse d'impact
        impact_df.to_csv(f"{output_dir}/{fig_count:02d}_impact_comparatif.csv", index=False)
        fig_count += 1
    
    # 9. Conclusions et observations
    print("\n=== Principales observations sur l'impact des hôtes et avis ===")
    
    # Corrélations avec le prix
    corr_with_price = corr_matrix['log_price'].drop('log_price').sort_values(ascending=False)
    print("\nCorrélation avec log_price (facteurs de réputation):")
    for factor, corr in corr_with_price.items():
        if not pd.isna(corr):
            print(f"- {factor}: {corr:.4f}")
    
    # Moyennes par catégorie de réputation
    rep_impact = df.groupby('reputation_category')['price'].mean().dropna()
    if len(rep_impact) > 0:
        print("\nPrix moyen par niveau de réputation:")
        for category, price in rep_impact.items():
            print(f"- {category}: ${price:.2f}")
    
    # Sauvegarder un résumé des observations
    with open(f"{output_dir}/{fig_count:02d}_resume_observations_reputation.txt", 'w', encoding='utf-8') as f:
        f.write("=== RÉSUMÉ DES OBSERVATIONS SUR L'IMPACT DE LA RÉPUTATION ===\n\n")
        
        f.write("Corrélation avec log_price (facteurs de réputation):\n")
        for factor, corr in corr_with_price.items():
            if not pd.isna(corr):
                f.write(f"- {factor}: {corr:.4f}\n")
        
        if len(rep_impact) > 0:
            f.write("\nPrix moyen par niveau de réputation:\n")
            for category, price in rep_impact.items():
                f.write(f"- {category}: ${price:.2f}\n")
        
        if impact_analysis:
            f.write("\nImpacts comparatifs (% de différence de prix):\n")
            for factor, impact in impact_analysis.items():
                f.write(f"- {factor}: {impact:+.1f}%\n")
        
        f.write(f"\nNombre total d'hôtes analysés: {len(df)}\n")
        f.write(f"Indice de réputation moyen: {df['reputation_index'].mean():.2f}/10\n")
    
    print(f"\nAnalyse terminée. {fig_count} visualisations enregistrées dans {output_dir}")
    return df

# Exécuter l'analyse
if __name__ == "__main__":
    file_path = "Data/Clean/train_partial_clean_standart.csv"
    output_dir = "Data/Visual/plot/Multiple/hotes_avis"
    df_analyzed = analyze_hosts_reviews_impact(file_path, output_dir)