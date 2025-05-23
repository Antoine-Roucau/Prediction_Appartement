import pandas as pd
import numpy as np

def add_computed_features(df):
    """
    Ajoute 3 nouvelles colonnes calculées au dataset :
    - localisation = 3*property_type + city + 2*neighbourhood
    - reputation = review_score**2 * sqrt(nb_reviews)
    - couchages = bedroom * (nb_bed + accommodate)
    """
    print("Ajout des variables calculées...")
    df_enhanced = df.copy()
    
    # 1. LOCALISATION = 3*property_type + city + 2*neighbourhood
    print("Calcul de la variable 'localisation'...")
    
    # Vérifier les colonnes nécessaires
    required_cols_localisation = ['property_type_encoded', 'city_encoded', 'neighbourhood_encoded']
    missing_cols = [col for col in required_cols_localisation if col not in df.columns]
    
    if missing_cols:
        print(f"⚠️  Colonnes manquantes pour localisation: {missing_cols}")
        # Utiliser les colonnes alternatives si les encodées n'existent pas
        property_type_col = 'property_type_encoded' if 'property_type_encoded' in df.columns else 'property_type'
        city_col = 'city_encoded' if 'city_encoded' in df.columns else 'city'
        neighbourhood_col = 'neighbourhood_encoded' if 'neighbourhood_encoded' in df.columns else 'neighbourhood'
        
        # Si les colonnes ne sont pas encodées, on doit d'abord les encoder
        if property_type_col == 'property_type':
            print("⚠️  Utilisation des colonnes non-encodées - résultat approximatif")
        
        df_enhanced['localisation'] = (3 * df_enhanced[property_type_col].fillna(0) + 
                                     df_enhanced[city_col].fillna(0) + 
                                     2 * df_enhanced[neighbourhood_col].fillna(0))
    else:
        df_enhanced['localisation'] = (3 * df_enhanced['property_type_encoded'].fillna(0) + 
                                     df_enhanced['city_encoded'].fillna(0) + 
                                     2 * df_enhanced['neighbourhood_encoded'].fillna(0))
    
    # 2. REPUTATION = review_score**2 * sqrt(nb_reviews)
    print("Calcul de la variable 'reputation'...")
    
    # Identifier les colonnes de review
    review_score_col = None
    nb_reviews_col = 1
    
    # Chercher les colonnes de scores et nombre de reviews
    for col in df.columns:
        if 'review_scores_rating' in col.lower():
            review_score_col = col
        elif 'number_of_reviews' in col.lower():
            nb_reviews_col = col
    
    if review_score_col is None or nb_reviews_col is None:
        print(f"⚠️  Colonnes de reviews non trouvées:")
        print(f"   - Score reviews: {review_score_col}")
        print(f"   - Nombre reviews: {nb_reviews_col}")
        print("   Colonnes disponibles:")
        review_cols = [col for col in df.columns if 'review' in col.lower()]
        for col in review_cols:
            print(f"     - {col}")
        
        # Utiliser les colonnes par défaut
        review_score_col = 'review_scores_rating' if 'review_scores_rating' in df.columns else review_cols[0] if review_cols else None
        nb_reviews_col = 'number_of_reviews' if 'number_of_reviews' in df.columns else review_cols[1] if len(review_cols) > 1 else None
    if review_score_col and nb_reviews_col:
        # Nettoyer les données
        review_scores = pd.to_numeric(df_enhanced[review_score_col], errors='coerce').fillna(0)
        nb_reviews = pd.to_numeric(df_enhanced[nb_reviews_col], errors='coerce').fillna(1)
        # Calculer la réputation
        # Normaliser les scores sur 100 si nécessaire
        if review_scores.max() <= 5:  # Échelle 1-5
            review_scores = review_scores * 20  # Convertir en échelle 0-100
        
        # Calculer: score²  * sqrt(nb_reviews)
        df_enhanced['reputation'] = (review_scores ** 2) * np.sqrt(nb_reviews)
        
        # Gérer les cas particuliers
        df_enhanced['reputation'] = df_enhanced['reputation'].fillna(0)
        df_enhanced['reputation'] = np.where(df_enhanced['reputation'] < 0, 0, df_enhanced['reputation'])
        
    else:
        print("❌ Impossible de calculer la réputation - colonnes manquantes")
        df_enhanced['reputation'] = 0
    
    # 3. COUCHAGES = bedroom * (nb_bed + accommodate)
    print("Calcul de la variable 'couchages'...")
    
    # Identifier les colonnes de couchage
    bedroom_col = None
    beds_col = None
    accommodate_col = None
    
    # Chercher les colonnes
    for col in df.columns:
        if 'bedroom' in col.lower():
            bedroom_col = col
        elif 'beds' in col.lower() and 'bedroom' not in col.lower():
            beds_col = col
        elif 'accommodate' in col.lower():
            accommodate_col = col
    
    if not all([bedroom_col, beds_col, accommodate_col]):
        print(f"⚠️  Colonnes de couchage détectées:")
        print(f"   - Bedrooms: {bedroom_col}")
        print(f"   - Beds: {beds_col}")
        print(f"   - Accommodates: {accommodate_col}")
        
        # Utiliser les colonnes par défaut
        bedroom_col = bedroom_col or 'bedrooms'
        beds_col = beds_col or 'beds'
        accommodate_col = accommodate_col or 'accommodates'
    
    # Calculer les couchages
    try:
        bedrooms = pd.to_numeric(df_enhanced[bedroom_col], errors='coerce').fillna(1)
        beds = pd.to_numeric(df_enhanced[beds_col], errors='coerce').fillna(1)
        accommodates = pd.to_numeric(df_enhanced[accommodate_col], errors='coerce').fillna(2)
        
        # Calculer: bedroom * (nb_bed + accommodate)
        df_enhanced['couchages'] = bedrooms * (beds + accommodates)
        
        # Gérer les valeurs aberrantes
        df_enhanced['couchages'] = np.where(df_enhanced['couchages'] < 0, 0, df_enhanced['couchages'])
        df_enhanced['couchages'] = np.where(df_enhanced['couchages'] > 50, 50, df_enhanced['couchages'])  # Cap à 50
        
    except Exception as e:
        print(f"❌ Erreur dans le calcul des couchages: {e}")
        df_enhanced['couchages'] = 0
    
    # Statistiques des nouvelles variables
    print("\n=== STATISTIQUES DES NOUVELLES VARIABLES ===")
    new_vars = ['localisation', 'reputation', 'couchages']
    
    for var in new_vars:
        if var in df_enhanced.columns:
            stats = df_enhanced[var].describe()
            print(f"\n{var.upper()}:")
            print(f"  Min: {stats['min']:.2f}")
            print(f"  Max: {stats['max']:.2f}")
            print(f"  Moyenne: {stats['mean']:.2f}")
            print(f"  Médiane: {stats['50%']:.2f}")
            print(f"  Écart-type: {stats['std']:.2f}")
            
            # Vérifier les valeurs nulles
            null_count = df_enhanced[var].isnull().sum()
            if null_count > 0:
                print(f"  ⚠️  Valeurs nulles: {null_count}")
    
    print(f"\n✅ {len(new_vars)} nouvelles variables ajoutées au dataset")
    print(f"📊 Dataset final: {df_enhanced.shape[0]} lignes, {df_enhanced.shape[1]} colonnes")
    
    return df_enhanced

def validate_computed_features(df):
    """
    Valide les nouvelles variables calculées
    """
    print("\n=== VALIDATION DES VARIABLES CALCULÉES ===")
    
    # 1. Validation localisation
    if 'localisation' in df.columns:
        loc_stats = df['localisation'].describe()
        print(f"\nLOCALISATION:")
        print(f"  Plage théorique: 3-50 (3*1+1+2*1 à 3*10+5+2*5)")
        print(f"  Plage observée: {loc_stats['min']:.0f}-{loc_stats['max']:.0f}")
        
        # Vérifier la distribution
        print(f"  Distribution par quartiles:")
        for q in [0.25, 0.5, 0.75, 0.9]:
            val = df['localisation'].quantile(q)
            print(f"    Q{int(q*100)}: {val:.1f}")
    
    # 2. Validation réputation
    if 'reputation' in df.columns:
        rep_stats = df['reputation'].describe()
        print(f"\nREPUTATION:")
        print(f"  Formule: score² * sqrt(nb_reviews)")
        print(f"  Plage observée: {rep_stats['min']:.0f}-{rep_stats['max']:.0f}")
        
        # Cas particuliers
        zero_rep = (df['reputation'] == 0).sum()
        print(f"  Propriétés sans réputation (0): {zero_rep} ({zero_rep/len(df)*100:.1f}%)")
        
        high_rep = (df['reputation'] > 50000).sum()
        if high_rep > 0:
            print(f"  ⚠️  Propriétés avec réputation très élevée (>50k): {high_rep}")
    
    # 3. Validation couchages
    if 'couchages' in df.columns:
        couch_stats = df['couchages'].describe()
        print(f"\nCOUCHAGES:")
        print(f"  Formule: bedrooms * (beds + accommodates)")
        print(f"  Plage observée: {couch_stats['min']:.0f}-{couch_stats['max']:.0f}")
        
        # Distribution
        print(f"  Distribution:")
        print(f"    Petits logements (≤6): {(df['couchages'] <= 6).sum()} ({(df['couchages'] <= 6).sum()/len(df)*100:.1f}%)")
        print(f"    Logements moyens (7-15): {((df['couchages'] > 6) & (df['couchages'] <= 15)).sum()} ({((df['couchages'] > 6) & (df['couchages'] <= 15)).sum()/len(df)*100:.1f}%)")
        print(f"    Grands logements (>15): {(df['couchages'] > 15).sum()} ({(df['couchages'] > 15).sum()/len(df)*100:.1f}%)")

def create_enhanced_model_dataset(df):
    """
    Crée un dataset final avec toutes les variables pour la modélisation
    """
    print("\n=== CRÉATION DU DATASET FINAL POUR MODÉLISATION ===")
    
    # Variables de base
    base_vars = ['id', 'log_price']
    
    # Variables encodées
    encoded_vars = [col for col in df.columns if col.endswith('_encoded') or col.endswith('_numeric')]
    
    # Variables calculées
    computed_vars = ['localisation', 'reputation', 'couchages']
    computed_vars = [var for var in computed_vars if var in df.columns]
    
    # Variables spéciales
    special_vars = ['amenities_score', 'host_experience_years']
    special_vars = [var for var in special_vars if var in df.columns]
    
    # Combiner toutes les variables
    all_vars = base_vars + encoded_vars + computed_vars + special_vars
    
    # Filtrer les variables qui existent
    existing_vars = [var for var in all_vars if var in df.columns]
    
    # Créer le dataset final
    df_final = df[existing_vars].copy()
    
    # Nettoyer les valeurs manquantes
    df_final = df_final.fillna(0)
    
    print(f"Dataset final créé avec {len(existing_vars)} variables:")
    print(f"  - Variables de base: {len(base_vars)}")
    print(f"  - Variables encodées: {len([v for v in encoded_vars if v in existing_vars])}")
    print(f"  - Variables calculées: {len(computed_vars)}")
    print(f"  - Variables spéciales: {len([v for v in special_vars if v in existing_vars])}")
    
    return df_final

def main():
    """Fonction principale"""
    # Fichiers d'entrée et sortie
    input_file = "Data/Clean/test_model_ready.csv"  # Fichier encodé
    output_file_enhanced = "Data/Clean/test_enhanced.csv"
    output_file_final = "Data/Clean/test_final_model.csv"
    
    try:
        # Charger les données encodées
        print(f"📂 Chargement des données depuis {input_file}...")
        df = pd.read_csv(input_file)
        print(f"✅ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        
        # Afficher les colonnes disponibles
        print(f"\n📋 Colonnes disponibles:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # Ajouter les variables calculées
        df_enhanced = add_computed_features(df)
        
        # Valider les nouvelles variables
        validate_computed_features(df_enhanced)
        
        # Sauvegarder le dataset enrichi
        df_enhanced.to_csv(output_file_enhanced, index=False)
        print(f"\n💾 Dataset enrichi sauvegardé: {output_file_enhanced}")
        
        # Créer le dataset final pour modélisation
        df_final = create_enhanced_model_dataset(df_enhanced)
        df_final.to_csv(output_file_final, index=False)
        print(f"💾 Dataset final sauvegardé: {output_file_final}")
        
        print(f"\n🎉 TRAITEMENT TERMINÉ AVEC SUCCÈS!")
        print(f"📊 Résumé:")
        print(f"   - Propriétés traitées: {df.shape[0]:,}")
        print(f"   - Variables initiales: {df.shape[1]}")
        print(f"   - Variables finales: {df_final.shape[1]}")
        print(f"   - Nouvelles variables: localisation, reputation, couchages")
        
        print(f"\n📁 Fichiers créés:")
        print(f"   - {output_file_enhanced} (dataset complet enrichi)")
        print(f"   - {output_file_final} (prêt pour modélisation)")
        
    except FileNotFoundError:
        print(f"❌ Erreur: Fichier {input_file} non trouvé!")
        print("💡 Assurez-vous d'avoir d'abord exécuté le script d'encodage.")
        print("   Fichiers alternatifs à essayer:")
        alternative_files = [
            "Data/Clean/train_encoded_full.csv",
            "Data/Clean/train_partial_clean_standart.csv"
        ]
        for alt_file in alternative_files:
            print(f"   - {alt_file}")
            
    except Exception as e:
        print(f"❌ Erreur lors du traitement: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()