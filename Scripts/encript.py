import pandas as pd
import numpy as np
import re
from datetime import datetime

# MAPPINGS BASÉS SUR VOS DONNÉES RÉELLES
PROPERTY_TYPE_MAPPING = {
    "Loft": 10,
    "Condominium": 9,
    "Apartment": 7,
    "Townhouse": 6,
    "Other": 4,
    "House": 3,
    "Guesthouse": 1
}

ROOM_TYPE_MAPPING = {
    "Entire home/apt": 5,
    "Private room": 2,
    "Shared room": 1
}

BED_TYPE_MAPPING = {
    "Real Bed": 1,
    "Pull-out Sofa": -1,
    "Couch": -1,
    "Airbed": -1,
    "Futon": -1
}

AMENITY_SCORES = {
    "Family/kid friendly": 10,
    "Doorman": 10,
    "TV": 10,
    "Cable TV": 10,
    "Kitchen": 10,
    "Heating": 10,
    "Indoor fireplace": 10,
    "Gym": 10,
    "24-hour check-in": 10,
    "Wheelchair accessible": 10,
    "Self Check-In": 10,
    "Private entrance": 10,
    "Hair dryer": 10,
    "Washer": 9,
    "Dryer": 9,
    "Iron": 9,
    "Internet": 9,
    "Lockbox": 9,
    "Shampoo": 9,
    "Wireless Internet": 9,
    "Coffee maker": 9,
    "Elevator": 9,
    "Carbon monoxide detector": 8,
    "Laptop friendly workspace": 8,
    "Hangers": 8,
    "Essentials": 8,
    "Hot tub": 8,
    "Stove": 8,
    "Cooking basics": 8,
    "Dishes and silverware": 8,
    "Oven": 8,
    "Smoke detector": 8,
    "Buzzer/wireless intercom": 7,
    "Pets allowed": 7,
    "Refrigerator": 7,
    "Bed linens": 7,
    "Microwave": 7,
    "Fire extinguisher": 7,
    "Air conditioning": 7,
    "Safety card": 6,
    "Free parking on premises": 6,
    "Elevator in building": 6,
    "Hot water": 6,
    "First aid kit": 3,
    "Breakfast": 1,
    "Dog(s)": 1,
    "translation missing: en.hosting_amenity_50": 1,
    "Pets live on this property": 1,
    "translation missing: en.hosting_amenity_49": 1,
    "Lock on bedroom door": 1
}

CITY_MAPPING = {
    "SF": 5,
    "Boston": 5,
    "DC": 4,
    "NYC": 3,
    "LA": 2,
    "Chicago": 1
}

NEIGHBOURHOOD_MAPPINGS = {
    "NYC": {
        "Tribeca": 5, "Noho": 5, "Soho": 5, "Midtown": 5, "Union Square": 5,
        "West Village": 5, "Flatiron District": 5, "Battery Park City": 5,
        "Greenwich Village": 5, "Chelsea": 5, "Hudson Square": 5,
        "Times Square/Theatre District": 5, "Meatpacking District": 5,
        "Nolita": 5, "Gramercy Park": 5, "Midtown East": 5, "Gowanus": 5,
        "Little Italy": 5, "Hell's Kitchen": 5, "Kips Bay": 5, "Red Hook": 5,
        "East Village": 5, "Upper West Side": 4, "Murray Hill": 4,
        "Financial District": 4, "Carroll Gardens": 4, "Downtown Brooklyn": 4,
        "Brooklyn Heights": 4, "Upper East Side": 4, "Lower East Side": 4,
        "Cobble Hill": 4, "Alphabet City": 4, "Greenwood Heights": 4,
        "Fort Greene": 4, "Boerum Hill": 4, "Park Slope": 4, "Prospect Heights": 4,
        "Chinatown": 4, "Williamsburg": 4, "Greenpoint": 4, "Bayside": 4,
        "Kew Garden Hills": 4, "Clinton Hill": 4, "Forest Hills": 3,
        "Flatlands": 3, "Harlem": 3, "St. George": 3, "Windsor Terrace": 3,
        "The Rockaways": 3, "Wakefield": 3, "East Harlem": 3, "Tremont": 3,
        "Ditmars / Steinway": 3, "Morningside Heights": 3, "Kensington": 3,
        "Brooklyn Navy Yard": 3, "Astoria": 3, "Woodside": 3, "Long Island City": 3,
        "Lefferts Garden": 3, "Canarsie": 3, "Riverdale": 3, "Crown Heights": 3,
        "Bedford-Stuyvesant": 3, "Stapleton": 3, "Hamilton Heights": 2,
        "Sunnyside": 2, "Roosevelt Island": 2, "Brighton Beach": 2, "Inwood": 2,
        "Rego Park": 2, "Flatbush": 2, "Ozone Park": 2, "Flushing": 2,
        "Washington Heights": 2, "East New York": 2, "Tompkinsville": 2,
        "Bushwick": 2, "Gravesend": 2, "Midwood": 2, "Kingsbridge Heights": 2,
        "Maspeth": 2, "Bay Ridge": 2, "Sheepshead Bay": 2, "Jamaica": 2,
        "Mott Haven": 2, "East Flatbush": 2, "Corona": 1, "West Brighton": 1,
        "Sunset Park": 1, "Ridgewood": 1, "Middle Village": 1, "Brownsville": 1,
        "Jackson Heights": 1, "Elmhurst": 1, "Soundview": 1, "Eastchester": 1,
        "Richmond Hill": 1, "Bensonhurst": 1, "Concourse": 1, "Highbridge": 1,
        "Allerton": 1, "Bronxdale": 1, "Claremont": 1, "East Elmhurst": 1,
        "Woodhaven": 1, "Baychester": 1
    },
    "SF": {
        "Cole Valley": 5, "Financial District": 5, "Marina": 5, "Chinatown": 5,
        "Telegraph Hill": 5, "Fisherman's Wharf": 5, "Pacific Heights": 5,
        "Cow Hollow": 5, "South Beach": 5, "Russian Hill": 5, "The Castro": 4,
        "Haight-Ashbury": 4, "SoMa": 4, "Presidio Heights": 4,
        "Western Addition/NOPA": 4, "Oceanview": 4, "Potrero Hill": 4,
        "Nob Hill": 4, "Glen Park": 4, "Noe Valley": 3, "Richmond District": 3,
        "Mission District": 3, "Lower Haight": 3, "Twin Peaks": 3, "Sunnyside": 3,
        "Alamo Square": 3, "Duboce Triangle": 3, "Downtown": 3, "Bernal Heights": 2,
        "Mission Terrace": 2, "Hayes Valley": 2, "Tenderloin": 2, "Inner Sunset": 2,
        "Outer Sunset": 2, "Parkside": 2, "Diamond Heights": 2, "Union Square": 2,
        "North Beach": 2, "Portola": 1, "Excelsior": 1, "Ingleside": 1,
        "Balboa Terrace": 1, "Visitacion Valley": 1, "Bayview": 1, "Lakeshore": 1,
        "Crocker Amazon": 1
    },
    "DC": {
        "Judiciary Square": 5, "Georgetown": 5, "Downtown/Penn Quarter": 5,
        "West End": 5, "Capitol Hill": 5, "Kalorama": 5, "LeDroit Park": 5,
        "Mount Vernon Square": 5, "Logan Circle": 5, "Dupont Circle": 4,
        "U Street Corridor": 4, "Chevy Chase": 4, "Foggy Bottom": 4,
        "Adams Morgan": 4, "Burleith": 4, "Shaw": 4,
        "Near Northeast/H Street Corridor": 4, "Truxton Circle": 3, "Trinidad": 3,
        "Navy Yard": 3, "16th Street Heights": 3, "Cleveland Park": 3,
        "Southwest Waterfront": 3, "Woodley Park": 3, "Columbia Heights": 3,
        "Palisades": 3, "Barney Circle": 2, "Pleasant Plains": 2, "Glover Park": 2,
        "Brookland": 2, "Mount Pleasant": 2, "Carver Langston": 2, "Park View": 2,
        "Edgewood": 2, "Petworth": 1, "Brentwood": 1, "Bloomingdale": 1,
        "Kingman Park": 1, "Manor Park": 1, "Eckington": 1, "Brightwood": 1,
        "Anacostia": 1, "Twining": 1
    },
    "Boston": {
        "Downtown Crossing": 5, "Financial District": 5, "Chinatown": 5,
        "Charlestown": 5, "Back Bay": 5, "Beacon Hill": 4, "South End": 4,
        "North End": 4, "West End": 4, "Theater District": 3, "South Boston": 3,
        "Fenway/Kenmore": 3, "West Roxbury": 3, "Roxbury": 2, "Jamaica Plain": 2,
        "East Boston": 2, "Roslindale": 2, "Mission Hill": 1, "Allston-Brighton": 1,
        "Mattapan": 1, "Dorchester": 1, "Hyde Park": 1
    },
    "LA": {
        "Malibu": 5, "Manhattan Beach": 5, "Laurel Canyon": 5, "Marina Del Rey": 5,
        "Pacific Palisades": 5, "Beverly Hills": 5, "Venice": 5,
        "Bel Air/Beverly Crest": 5, "Hermosa Beach": 5, "Cahuenga Pass": 5,
        "Topanga": 5, "Arts District": 5, "Santa Monica": 5, "Downtown": 5,
        "Hollywood Hills": 5, "West Covina": 5, "Skid Row": 5, "Del Rey": 5,
        "East Los Angeles": 4, "West Los Angeles": 4, "Westwood": 4,
        "West Hollywood": 4, "Studio City": 4, "Atwater Village": 4,
        "Westchester/Playa Del Rey": 4, "Brentwood": 4, "Silver Lake": 4,
        "Rancho Palos Verdes": 4, "Pasadena": 4, "Redondo Beach": 4,
        "Sherman Oaks": 4, "Hollywood": 4, "Los Feliz": 4, "Culver City": 4,
        "Glassell Park": 4, "Mid-Wilshire": 4, "Mar Vista": 3, "Westlake": 3,
        "El Segundo": 3, "Elysian Valley": 3, "Long Beach": 3, "Westside": 3,
        "South Robertson": 3, "South Pasadena": 3, "Whittier": 3, "Encino": 3,
        "Altadena": 3, "East Hollywood": 3, "Echo Park": 3, "Mount Washington": 3,
        "Van Nuys": 3, "Burbank": 3, "Glendale": 3, "Highland Park": 2,
        "Valley Village": 2, "Eagle Rock": 2, "El Sereno": 2, "Rosemead": 2,
        "Tarzana": 2, "Temple City": 2, "South LA": 2, "Hawthorne": 2,
        "Mid-City": 2, "Northridge": 2, "Harbor Gateway": 2, "Canoga Park": 2,
        "Woodland Hills/Warner Center": 2, "North Hollywood": 2, "Reseda": 2,
        "Sunland/Tujunga": 2, "San Pedro": 2, "Toluca Lake": 1, "Torrance": 1,
        "Alhambra": 1, "Lincoln Heights": 1, "Palms": 1, "West Hills": 1,
        "Boyle Heights": 1, "Arcadia": 1, "West Adams": 1, "Lake Balboa": 1,
        "Winnetka": 1, "Valley Glen": 1, "Inglewood": 1, "Gardena": 1,
        "San Gabriel": 1, "Monterey Park": 1, "Lakewood": 1, "El Monte": 1
    },
    "Chicago": {
        "Roscoe Village": 5, "Wrigleyville": 5, "Old Town": 5,
        "West Loop/Greektown": 5, "Gold Coast": 5, "South Loop/Printers Row": 5,
        "Wicker Park": 5, "Lakeview": 5, "Lincoln Park": 5, "Logan Square": 4,
        "Loop": 4, "Bucktown": 4, "River North": 4, "Andersonville": 4,
        "West Town/Noble Square": 4, "Near North Side": 4, "Ukrainian Village": 4,
        "Streeterville": 3, "North Center": 3, "Lincoln Square": 3,
        "Humboldt Park": 3, "Hyde Park": 3, "Albany Park": 3, "Edgewater": 3,
        "Little Italy/UIC": 3, "Bronzeville": 3, "Uptown": 2, "Pilsen": 2,
        "Irving Park": 2, "Rogers Park": 2, "Bridgeport": 2, "Kenwood": 2,
        "West Ridge": 2, "South Shore": 1, "Portage Park": 1, "Jefferson Park": 1,
        "Little Village": 1, "Garfield Park": 1, "Near West Side": 1,
        "Woodlawn": 1, "Avondale": 1
    }
}

CANCELLATION_MAPPING = {
    "super_strict_60": -2,
    "super_strict_30": -1,
    "strict": 1,
    "moderate": 2,
    "flexible": 3
}

def encode_airbnb_features(df):
    """
    Encode les variables catégorielles en valeurs numériques basées sur l'analyse réelle des données
    """
    print("Début de l'encodage des caractéristiques Airbnb...")
    df_encoded = df.copy()
    
    # 1. PROPERTY_TYPE
    print("Encodage property_type...")
    df_encoded['property_type_encoded'] = df_encoded['property_type'].map(PROPERTY_TYPE_MAPPING)
    df_encoded['property_type_encoded'] = df_encoded['property_type_encoded'].fillna(4)  # Valeur par défaut "Other"
    
    # 2. ROOM_TYPE
    print("Encodage room_type...")
    df_encoded['room_type_encoded'] = df_encoded['room_type'].map(ROOM_TYPE_MAPPING)
    df_encoded['room_type_encoded'] = df_encoded['room_type_encoded'].fillna(2)  # Valeur par défaut "Private room"
    
    # 3. BED_TYPE
    print("Encodage bed_type...")
    df_encoded['bed_type_encoded'] = df_encoded['bed_type'].map(BED_TYPE_MAPPING)
    df_encoded['bed_type_encoded'] = df_encoded['bed_type_encoded'].fillna(1)  # Valeur par défaut "Real Bed"
    
    # 4. CITY
    print("Encodage city...")
    df_encoded['city_encoded'] = df_encoded['city'].map(CITY_MAPPING)
    df_encoded['city_encoded'] = df_encoded['city_encoded'].fillna(3)  # Valeur médiane
    
    # 5. NEIGHBOURHOOD
    print("Encodage neighbourhood...")
    def encode_neighbourhood(row):
        city = row['city']
        neighbourhood = row['neighbourhood']
        
        if city in NEIGHBOURHOOD_MAPPINGS and neighbourhood in NEIGHBOURHOOD_MAPPINGS[city]:
            return NEIGHBOURHOOD_MAPPINGS[city][neighbourhood]
        return 3  # Valeur par défaut
    
    df_encoded['neighbourhood_encoded'] = df_encoded.apply(encode_neighbourhood, axis=1)
    
    # 6. CANCELLATION_POLICY
    print("Encodage cancellation_policy...")
    df_encoded['cancellation_policy_encoded'] = df_encoded['cancellation_policy'].map(CANCELLATION_MAPPING)
    df_encoded['cancellation_policy_encoded'] = df_encoded['cancellation_policy_encoded'].fillna(1)  # Valeur par défaut "strict"
    
    # 7. AMENITIES SCORE
    print("Calcul du score d'aménités...")
    def calculate_amenity_score(amenities_str):
        """Calcule le score total des aménités"""
        if pd.isna(amenities_str) or amenities_str == '':
            return 0
            
        try:
            # Nettoyer la chaîne d'aménités
            if amenities_str.startswith('{') and amenities_str.endswith('}'):
                amenities_str = amenities_str[1:-1]
            
            # Extraire les aménités
            amenities_list = []
            items = amenities_str.split(',')
            for item in items:
                clean_item = item.strip().strip('"').strip("'")
                if clean_item and clean_item != '':
                    amenities_list.append(clean_item)
            
            total_score = 0
            matched_count = 0
            
            for amenity in amenities_list:
                if amenity in AMENITY_SCORES:
                    total_score += AMENITY_SCORES[amenity]
                    matched_count += 1
                else:
                    # Score par défaut pour les aménités non mappées
                    total_score += 2
            
            return total_score
            
        except Exception as e:
            print(f"Erreur dans le calcul des aménités: {e}")
            return 0
    
    df_encoded['amenities_score'] = df_encoded['amenities'].apply(calculate_amenity_score)
    
    # 8. HOST_EXPERIENCE_YEARS
    print("Calcul de l'expérience des hôtes...")
    def calculate_host_experience(host_since_str):
        """Calcule l'expérience de l'hôte en années"""
        if pd.isna(host_since_str):
            return 0.0
            
        try:
            # Convertir en datetime
            host_since = pd.to_datetime(host_since_str)
            
            # Date de référence basée sur vos données (2017)
            reference_date = pd.to_datetime('2017-01-01')
            
            # Calculer la différence en années
            experience_years = (reference_date - host_since).days / 365.25
            
            # Limiter entre 0 et 15 ans
            return max(0.0, min(15.0, experience_years))
            
        except Exception as e:
            return 0.0
    
    df_encoded['host_experience_years'] = df_encoded['host_since'].apply(calculate_host_experience)
    
    # 9. VARIABLES BOOLÉENNES
    print("Encodage des variables booléennes...")
    boolean_vars = ['cleaning_fee', 'instant_bookable', 'host_identity_verified']
    
    for var in boolean_vars:
        if var in df_encoded.columns:
            df_encoded[f'{var}_encoded'] = df_encoded[var].map({
                True: 1, 'True': 1, 't': 1, 'T': 1,
                False: 0, 'False': 0, 'f': 0, 'F': 0
            }).fillna(0).astype(int)
    
    # 10. VARIABLES NUMÉRIQUES (garder telles quelles)
    numeric_vars = ['accommodates', 'bedrooms', 'beds', 'bathrooms', 
                   'number_of_reviews', 'review_scores_rating', 'latitude', 'longitude']
    
    for var in numeric_vars:
        if var in df_encoded.columns:
            df_encoded[f'{var}_numeric'] = pd.to_numeric(df_encoded[var], errors='coerce').fillna(0)
    
    # Résumé de l'encodage
    print("\n=== RÉSUMÉ DE L'ENCODAGE ===")
    encoded_vars = [col for col in df_encoded.columns if col.endswith('_encoded') or col.endswith('_score') or col.endswith('_years') or col.endswith('_numeric')]
    print(f"Variables encodées créées: {len(encoded_vars)}")
    for var in encoded_vars:
        print(f"- {var}")
    
    # Statistiques de quelques variables clés
    print(f"\nStatistiques des variables encodées:")
    key_vars = ['property_type_encoded', 'room_type_encoded', 'city_encoded', 
               'amenities_score', 'host_experience_years']
    for var in key_vars:
        if var in df_encoded.columns:
            print(f"{var}: min={df_encoded[var].min():.2f}, max={df_encoded[var].max():.2f}, mean={df_encoded[var].mean():.2f}")
    
    return df_encoded

def create_model_ready_dataset(df_encoded):
    """
    Crée un dataset prêt pour la modélisation avec seulement les variables encodées
    """
    # Variables à conserver pour la modélisation
    model_vars = [
        'id', 'log_price',  # Variables cibles
        'property_type_encoded', 'room_type_encoded', 'bed_type_encoded',
        'city_encoded', 'neighbourhood_encoded', 'cancellation_policy_encoded',
        'amenities_score', 'host_experience_years',
        'cleaning_fee_encoded', 'instant_bookable_encoded', 'host_identity_verified_encoded',
        'accommodates_numeric', 'bedrooms_numeric', 'beds_numeric', 'bathrooms_numeric',
        'number_of_reviews_numeric', 'review_scores_rating_numeric',
        'latitude_numeric', 'longitude_numeric'
    ]
    
    # Filtrer les variables qui existent dans le DataFrame
    existing_vars = [var for var in model_vars if var in df_encoded.columns]
    
    df_model = df_encoded[existing_vars].copy()
    
    # Remplacer les valeurs manquantes
    df_model = df_model.fillna(0)
    
    print(f"\nDataset pour modélisation créé avec {len(existing_vars)} variables:")
    for var in existing_vars:
        print(f"- {var}")
    
    return df_model

def main():
    """Fonction principale"""
    input_file = "Data/Clean/test_partial_clean_standart.csv"
    output_file_full = "Data/Clean/test_encoded_full.csv"
    output_file_model = "Data/Clean/test_model_ready.csv"
    
    try:
        # Charger les données
        print(f"Chargement des données depuis {input_file}...")
        df = pd.read_csv(input_file)
        print(f"Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        
        # Encoder les variables
        df_encoded = encode_airbnb_features(df)
        
        # Sauvegarder le dataset complet avec toutes les colonnes
        df_encoded.to_csv(output_file_full, index=False)
        print(f"\nDataset complet sauvegardé: {output_file_full}")
        
        # Créer et sauvegarder le dataset prêt pour la modélisation
        df_model = create_model_ready_dataset(df_encoded)
        df_model.to_csv(output_file_model, index=False)
        print(f"Dataset pour modélisation sauvegardé: {output_file_model}")
        
        print(f"\n✅ ENCODAGE TERMINÉ AVEC SUCCÈS!")
        print(f"📊 {df.shape[0]} propriétés encodées")
        print(f"📁 Fichiers créés:")
        print(f"   - {output_file_full} (dataset complet)")
        print(f"   - {output_file_model} (prêt pour modélisation)")
        
    except FileNotFoundError:
        print(f"❌ Erreur: Fichier {input_file} non trouvé!")
        print("Vérifiez le chemin du fichier.")
    except Exception as e:
        print(f"❌ Erreur lors de l'encodage: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()