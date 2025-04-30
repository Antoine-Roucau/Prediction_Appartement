import pandas as pd

def standardize_property_types(input_file_path, output_file_path):
    """
    Standardise les types de propriétés en gardant seulement certaines catégories 
    et en regroupant les autres dans 'Other'
    
    Args:
        input_file_path: Chemin du fichier CSV source
        output_file_path: Chemin du fichier CSV de sortie
    """
    try:
        # Lecture du fichier CSV
        print(f"Lecture du fichier {input_file_path}...")
        df = pd.read_csv(input_file_path)
        
        # Afficher la distribution originale des types de propriétés
        print("\nDistribution originale des types de propriétés:")
        print(df['property_type'].value_counts())
        
        # Liste des types de propriétés à conserver
        property_types_to_keep = [
            'Apartment',
            'House',
            'Condominium',
            'Townhouse',
            'Loft',
            'Guesthouse'
        ]
        
        # Standardiser les types de propriétés
        print("\nStandardisation des types de propriétés...")
        df['property_type'] = df['property_type'].apply(
            lambda x: x if x in property_types_to_keep else 'Other'
        )
        
        # Afficher la nouvelle distribution
        print("\nNouvelle distribution des types de propriétés:")
        print(df['property_type'].value_counts())
        
        # Enregistrement du fichier modifié
        print(f"\nEnregistrement du fichier modifié vers {output_file_path}...")
        df.to_csv(output_file_path, index=False)
        
        print("Modification terminée avec succès!")
        return df
        
    except Exception as e:
        print(f"Une erreur est survenue: {str(e)}")
        return None

# Utilisation du script
if __name__ == "__main__":
    input_file = "Data/Clean/train_partial_clean.csv"  # CSV original
    output_file = "Data/Clean/train_partial_clean_standart.csv"  # CSV clean
    
    df_modified = standardize_property_types(input_file, output_file)