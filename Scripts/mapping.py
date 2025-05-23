import pandas as pd
import numpy as np
import re
from datetime import datetime
import json

def analyze_airbnb_data_for_encoding(file_path):
    """
    Analyse complète des données Airbnb pour générer les mappings d'encodage optimaux
    """
    print("Chargement des données...")
    df = pd.read_csv(file_path)
    print(f"Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # Convertir log_price en price pour l'analyse
    df['price'] = np.exp(df['log_price'])
    
    results = {}
    
    # ===== 1. ANALYSIS PROPERTY_TYPE =====
    print("\n" + "="*50)
    print("1. ANALYSE DES TYPES DE PROPRIÉTÉS")
    print("="*50)
    
    property_analysis = df.groupby('property_type').agg({
        'price': ['count', 'mean', 'median', 'std'],
        'log_price': ['mean', 'median']
    }).round(2)
    
    property_analysis.columns = ['count', 'price_mean', 'price_median', 'price_std', 'log_price_mean', 'log_price_median']
    property_analysis = property_analysis.sort_values('price_median', ascending=False)
    
    print("Analyse des types de propriétés (trié par prix médian décroissant):")
    print(property_analysis)
    
    # Générer le mapping 1-10
    property_types = property_analysis.index.tolist()
    property_mapping = {}
    for i, prop_type in enumerate(property_types):
        # Score de 10 (le plus cher) à 1 (le moins cher)
        score = max(1, 10 - int(i * 9 / (len(property_types) - 1)))
        property_mapping[prop_type] = score
    
    results['property_type_mapping'] = property_mapping
    
    # ===== 2. ANALYSIS ROOM_TYPE =====
    print("\n" + "="*50)
    print("2. ANALYSE DES TYPES DE CHAMBRES")
    print("="*50)
    
    room_analysis = df.groupby('room_type').agg({
        'price': ['count', 'mean', 'median', 'std'],
        'log_price': ['mean', 'median']
    }).round(2)
    
    room_analysis.columns = ['count', 'price_mean', 'price_median', 'price_std', 'log_price_mean', 'log_price_median']
    room_analysis = room_analysis.sort_values('price_median', ascending=False)
    
    print("Analyse des types de chambres:")
    print(room_analysis)
    
    # Mapping room_type
    room_mapping = {}
    room_types = room_analysis.index.tolist()
    for i, room_type in enumerate(room_types):
        if 'Entire' in room_type:
            room_mapping[room_type] = 5
        elif 'Private' in room_type:
            room_mapping[room_type] = 2
        elif 'Shared' in room_type:
            room_mapping[room_type] = 1
        else:
            room_mapping[room_type] = 3  # Default
    
    results['room_type_mapping'] = room_mapping
    
    # ===== 3. ANALYSIS BED_TYPE =====
    print("\n" + "="*50)
    print("3. ANALYSE DES TYPES DE LITS")
    print("="*50)
    
    bed_analysis = df.groupby('bed_type').agg({
        'price': ['count', 'mean', 'median', 'std'],
        'log_price': ['mean', 'median']
    }).round(2)
    
    bed_analysis.columns = ['count', 'price_mean', 'price_median', 'price_std', 'log_price_mean', 'log_price_median']
    bed_analysis = bed_analysis.sort_values('price_median', ascending=False)
    
    print("Analyse des types de lits:")
    print(bed_analysis)
    
    # Mapping bed_type
    bed_mapping = {}
    bed_types = bed_analysis.index.tolist()
    real_bed_price = bed_analysis.loc[bed_analysis.index.str.contains('Real', case=False, na=False), 'price_median'].iloc[0] if any(bed_analysis.index.str.contains('Real', case=False, na=False)) else bed_analysis['price_median'].max()
    
    for bed_type in bed_types:
        if 'Real' in bed_type:
            bed_mapping[bed_type] = 1
        else:
            # Score négatif proportionnel à l'écart avec Real Bed
            bed_price = bed_analysis.loc[bed_type, 'price_median']
            penalty = int((real_bed_price - bed_price) / real_bed_price * 4)  # -1 à -4
            bed_mapping[bed_type] = -max(1, penalty)
    
    results['bed_type_mapping'] = bed_mapping
    
    # ===== 4. ANALYSIS AMENITIES =====
    print("\n" + "="*50)
    print("4. ANALYSE DES AMÉNITÉS")
    print("="*50)
    
    def extract_amenities_from_string(amenities_str):
        """Extrait les aménités de la chaîne"""
        if pd.isna(amenities_str) or amenities_str == '':
            return []
            
        try:
            if amenities_str.startswith('{') and amenities_str.endswith('}'):
                amenities_str = amenities_str[1:-1]
            
            amenities_list = []
            items = amenities_str.split(',')
            for item in items:
                clean_item = item.strip().strip('"').strip("'")
                if clean_item and clean_item != '':
                    amenities_list.append(clean_item)
            
            return amenities_list
        except:
            return []
    
    # Extraire toutes les aménités
    all_amenities = []
    for amenities_str in df['amenities']:
        all_amenities.extend(extract_amenities_from_string(amenities_str))
    
    # Compter les occurrences
    amenity_counts = pd.Series(all_amenities).value_counts()
    print(f"Nombre total d'aménités uniques: {len(amenity_counts)}")
    print(f"Top 30 aménités les plus fréquentes:")
    print(amenity_counts.head(30))
    
    # Analyser l'impact de chaque aménité sur les prix
    print("\nAnalyse de l'impact des aménités sur les prix...")
    
    amenity_impact = {}
    top_amenities = amenity_counts.head(50).index.tolist()  # Analyser les 50 plus fréquentes
    
    for amenity in top_amenities:
        # Créer une colonne binaire pour cette aménité
        df[f'has_{amenity}'] = df['amenities'].apply(
            lambda x: 1 if amenity in str(x) else 0
        )
        
        # Calculer les prix avec et sans cette aménité
        with_amenity = df[df[f'has_{amenity}'] == 1]['price'].median()
        without_amenity = df[df[f'has_{amenity}'] == 0]['price'].median()
        count_with = df[f'has_{amenity}'].sum()
        
        if without_amenity > 0 and count_with >= 10:  # Au moins 10 occurrences
            impact_pct = (with_amenity - without_amenity) / without_amenity * 100
            
            amenity_impact[amenity] = {
                'count': count_with,
                'price_with': with_amenity,
                'price_without': without_amenity,
                'impact_pct': impact_pct
            }
    
    # Trier par impact
    amenity_impact_df = pd.DataFrame(amenity_impact).T
    amenity_impact_df = amenity_impact_df.sort_values('impact_pct', ascending=False)
    
    print("Top 20 aménités par impact sur les prix:")
    print(amenity_impact_df.head(20))
    
    # Générer les scores d'aménités (1-10)
    amenity_scores = {}
    for amenity, data in amenity_impact_df.iterrows():
        impact = data['impact_pct']
        if impact >= 25:
            score = 10
        elif impact >= 20:
            score = 9
        elif impact >= 15:
            score = 8
        elif impact >= 10:
            score = 7
        elif impact >= 5:
            score = 6
        elif impact >= 2:
            score = 5
        elif impact >= 0:
            score = 4
        elif impact >= -5:
            score = 3
        elif impact >= -10:
            score = 2
        else:
            score = 1
            
        amenity_scores[amenity] = score
    
    results['amenity_scores'] = amenity_scores
    results['amenity_impact_analysis'] = amenity_impact_df
    
    # ===== 5. ANALYSIS CITY =====
    print("\n" + "="*50)
    print("5. ANALYSE DES VILLES")
    print("="*50)
    
    city_analysis = df.groupby('city').agg({
        'price': ['count', 'mean', 'median', 'std'],
        'log_price': ['mean', 'median']
    }).round(2)
    
    city_analysis.columns = ['count', 'price_mean', 'price_median', 'price_std', 'log_price_mean', 'log_price_median']
    city_analysis = city_analysis.sort_values('price_median', ascending=False)
    
    print("Analyse des villes:")
    print(city_analysis)
    
    # Mapping city (1-5)
    cities = city_analysis.index.tolist()
    city_mapping = {}
    for i, city in enumerate(cities):
        score = max(1, 5 - int(i * 4 / (len(cities) - 1))) if len(cities) > 1 else 5
        city_mapping[city] = score
    
    results['city_mapping'] = city_mapping
    
    # ===== 6. ANALYSIS NEIGHBOURHOOD par ville =====
    print("\n" + "="*50)
    print("6. ANALYSE DES QUARTIERS PAR VILLE")
    print("="*50)
    
    neighbourhood_mappings = {}
    
    for city in df['city'].unique():
        print(f"\n--- Analyse des quartiers pour {city} ---")
        city_df = df[df['city'] == city]
        
        neighbourhood_analysis = city_df.groupby('neighbourhood').agg({
            'price': ['count', 'mean', 'median'],
            'log_price': 'median'
        }).round(2)
        
        neighbourhood_analysis.columns = ['count', 'price_mean', 'price_median', 'log_price_median']
        
        # Filtrer les quartiers avec au moins 5 propriétés
        neighbourhood_analysis = neighbourhood_analysis[neighbourhood_analysis['count'] >= 5]
        neighbourhood_analysis = neighbourhood_analysis.sort_values('price_median', ascending=False)
        
        print(f"Top 15 quartiers pour {city}:")
        print(neighbourhood_analysis.head(15))
        
        # Créer le mapping pour cette ville
        neighbourhoods = neighbourhood_analysis.index.tolist()
        city_neighbourhood_mapping = {}
        
        if len(neighbourhoods) > 0:
            # Calculer les quantiles
            q80 = neighbourhood_analysis['price_median'].quantile(0.8)
            q60 = neighbourhood_analysis['price_median'].quantile(0.6)
            q40 = neighbourhood_analysis['price_median'].quantile(0.4)
            q20 = neighbourhood_analysis['price_median'].quantile(0.2)
            
            for neighbourhood in neighbourhoods:
                price = neighbourhood_analysis.loc[neighbourhood, 'price_median']
                if price >= q80:
                    score = 5
                elif price >= q60:
                    score = 4
                elif price >= q40:
                    score = 3
                elif price >= q20:
                    score = 2
                else:
                    score = 1
                
                city_neighbourhood_mapping[neighbourhood] = score
        
        neighbourhood_mappings[city] = city_neighbourhood_mapping
    
    results['neighbourhood_mappings'] = neighbourhood_mappings
    
    # ===== 7. ANALYSIS CANCELLATION_POLICY =====
    print("\n" + "="*50)
    print("7. ANALYSE DES POLITIQUES D'ANNULATION")
    print("="*50)
    
    cancellation_analysis = df.groupby('cancellation_policy').agg({
        'price': ['count', 'mean', 'median', 'std'],
        'log_price': ['mean', 'median']
    }).round(2)
    
    cancellation_analysis.columns = ['count', 'price_mean', 'price_median', 'price_std', 'log_price_mean', 'log_price_median']
    cancellation_analysis = cancellation_analysis.sort_values('price_median', ascending=False)
    
    print("Analyse des politiques d'annulation:")
    print(cancellation_analysis)
    
    # Mapping cancellation_policy
    cancellation_mapping = {}
    policies = cancellation_analysis.index.tolist()
    for i, policy in enumerate(policies):
        score = max(1, 3 - int(i * 2 / (len(policies) - 1))) if len(policies) > 1 else 3
        cancellation_mapping[policy] = score
    
    results['cancellation_mapping'] = cancellation_mapping
    
    # ===== 8. ANALYSIS AUTRES VARIABLES BOOLÉENNES =====
    print("\n" + "="*50)
    print("8. ANALYSE DES VARIABLES BOOLÉENNES")
    print("="*50)
    
    boolean_vars = ['cleaning_fee', 'instant_bookable', 'host_identity_verified']
    
    for var in boolean_vars:
        if var in df.columns:
            print(f"\n--- Analyse de {var} ---")
            
            # Normaliser les valeurs booléennes
            df[f'{var}_normalized'] = df[var].map({
                True: 1, 'True': 1, 't': 1, 'T': 1,
                False: 0, 'False': 0, 'f': 0, 'F': 0
            }).fillna(0)
            
            bool_analysis = df.groupby(f'{var}_normalized').agg({
                'price': ['count', 'mean', 'median'],
                'log_price': 'median'
            }).round(2)
            
            print(bool_analysis)
    
    # ===== 9. ANALYSIS HOST_SINCE =====
    print("\n" + "="*50)
    print("9. ANALYSE DE L'EXPÉRIENCE DES HÔTES")
    print("="*50)
    
    # Calculer l'expérience des hôtes
    df['host_since_date'] = pd.to_datetime(df['host_since'], errors='coerce')
    reference_date = pd.to_datetime('2017-01-01')  # Adapter selon vos données
    df['host_experience_years'] = (reference_date - df['host_since_date']).dt.days / 365.25
    df['host_experience_years'] = df['host_experience_years'].clip(0, 15)
    
    # Analyser la corrélation avec les prix
    experience_corr = df[['host_experience_years', 'price', 'log_price']].corr()
    print("Corrélation expérience hôte vs prix:")
    print(experience_corr)
    
    # Analyse par catégories d'expérience
    df['experience_category'] = pd.cut(df['host_experience_years'], 
                                     bins=[0, 1, 3, 6, 15], 
                                     labels=['Nouveau (0-1)', 'Junior (1-3)', 'Expérimenté (3-6)', 'Vétéran (6+)'])
    
    experience_analysis = df.groupby('experience_category').agg({
        'price': ['count', 'mean', 'median'],
        'log_price': 'median'
    }).round(2)
    
    print("Analyse par catégorie d'expérience:")
    print(experience_analysis)
    
    return results

def save_analysis_results(results, output_file='encoding_mappings.json'):
    """Sauvegarde tous les résultats d'analyse"""
    
    # Convertir les DataFrames en dictionnaires pour la sérialisation JSON
    results_serializable = {}
    
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            results_serializable[key] = value.to_dict()
        elif isinstance(value, dict):
            # Vérifier si c'est un dictionnaire de DataFrames
            serializable_dict = {}
            for k, v in value.items():
                if isinstance(v, pd.DataFrame):
                    serializable_dict[k] = v.to_dict()
                else:
                    serializable_dict[k] = v
            results_serializable[key] = serializable_dict
        else:
            results_serializable[key] = value
    
    # Sauvegarder en JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\nRésultats sauvegardés dans {output_file}")

def generate_encoding_script(results, output_file='generated_encoding_script.py'):
    """Génère automatiquement le script d'encodage basé sur l'analyse"""
    
    script_content = f'''# Script d'encodage généré automatiquement
import pandas as pd
import numpy as np
import re
from datetime import datetime

# MAPPINGS GÉNÉRÉS AUTOMATIQUEMENT
PROPERTY_TYPE_MAPPING = {results['property_type_mapping']}

ROOM_TYPE_MAPPING = {results['room_type_mapping']}

BED_TYPE_MAPPING = {results['bed_type_mapping']}

AMENITY_SCORES = {dict(list(results['amenity_scores'].items())[:50])}  # Top 50

CITY_MAPPING = {results['city_mapping']}

NEIGHBOURHOOD_MAPPINGS = {results['neighbourhood_mappings']}

CANCELLATION_MAPPING = {results['cancellation_mapping']}

def encode_features(df):
    """Encode les features selon l'analyse"""
    df_encoded = df.copy()
    
    # Property type
    df_encoded['property_type_encoded'] = df_encoded['property_type'].map(PROPERTY_TYPE_MAPPING).fillna(5)
    
    # Room type
    df_encoded['room_type_encoded'] = df_encoded['room_type'].map(ROOM_TYPE_MAPPING).fillna(2)
    
    # Bed type
    df_encoded['bed_type_encoded'] = df_encoded['bed_type'].map(BED_TYPE_MAPPING).fillna(1)
    
    # City
    df_encoded['city_encoded'] = df_encoded['city'].map(CITY_MAPPING).fillna(3)
    
    # Cancellation policy
    df_encoded['cancellation_policy_encoded'] = df_encoded['cancellation_policy'].map(CANCELLATION_MAPPING).fillna(2)
    
    # Neighbourhood
    def encode_neighbourhood(row):
        city = row['city']
        neighbourhood = row['neighbourhood']
        if city in NEIGHBOURHOOD_MAPPINGS and neighbourhood in NEIGHBOURHOOD_MAPPINGS[city]:
            return NEIGHBOURHOOD_MAPPINGS[city][neighbourhood]
        return 3
    
    df_encoded['neighbourhood_encoded'] = df_encoded.apply(encode_neighbourhood, axis=1)
    
    # Amenities score
    def calculate_amenity_score(amenities_str):
        if pd.isna(amenities_str):
            return 0
        
        total_score = 0
        try:
            if amenities_str.startswith('{{') and amenities_str.endswith('}}'):
                amenities_str = amenities_str[1:-1]
            
            items = amenities_str.split(',')
            for item in items:
                clean_item = item.strip().strip('"').strip("'")
                if clean_item in AMENITY_SCORES:
                    total_score += AMENITY_SCORES[clean_item]
                else:
                    total_score += 1  # Score par défaut
        except:
            pass
        
        return total_score
    
    df_encoded['amenities_score'] = df_encoded['amenities'].apply(calculate_amenity_score)
    
    # Host experience
    df_encoded['host_since_date'] = pd.to_datetime(df_encoded['host_since'], errors='coerce')
    reference_date = pd.to_datetime('2017-01-01')
    df_encoded['host_experience_years'] = (reference_date - df_encoded['host_since_date']).dt.days / 365.25
    df_encoded['host_experience_years'] = df_encoded['host_experience_years'].clip(0, 15).fillna(0)
    
    # Boolean variables
    boolean_vars = ['cleaning_fee', 'instant_bookable', 'host_identity_verified']
    for var in boolean_vars:
        if var in df_encoded.columns:
            df_encoded[f'{{var}}_encoded'] = df_encoded[var].map({{
                True: 1, 'True': 1, 't': 1, 'T': 1,
                False: 0, 'False': 0, 'f': 0, 'F': 0
            }}).fillna(0)
    
    return df_encoded

if __name__ == "__main__":
    df = pd.read_csv("Data/Clean/train_partial_clean_standart.csv")
    df_encoded = encode_features(df)
    df_encoded.to_csv("Data/Clean/train_encoded_auto.csv", index=False)
    print("Encodage terminé!")
'''
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"Script d'encodage généré: {output_file}")

# UTILISATION
if __name__ == "__main__":
    file_path = "Data/Clean/train_partial_clean_standart.csv"
    
    try:
        # Analyser les données
        results = analyze_airbnb_data_for_encoding(file_path)
        
        # Sauvegarder les résultats
        save_analysis_results(results)
        
        # Générer le script d'encodage
        generate_encoding_script(results)
        
        print("\n" + "="*50)
        print("ANALYSE TERMINÉE!")
        print("Fichiers générés:")
        print("- encoding_mappings.json (résultats complets)")
        print("- generated_encoding_script.py (script d'encodage)")
        print("="*50)
        
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()