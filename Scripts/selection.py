import pandas as pd

selected_columns = [
    # Id et prix
    'id', # 'log_price', pour le fichier train
    # Appartement
    'property_type', 'room_type', 'accommodates', 'bedrooms', 
    'beds','bed_type', 'bathrooms', 'amenities',
    # Loc
    'city', 'neighbourhood', 'latitude', 'longitude',
    # Règles et politiques
    'cancellation_policy', 'cleaning_fee', 'instant_bookable',
    # Host
    'host_since', 'host_identity_verified',
    # Reviews
    'number_of_reviews', 'review_scores_rating'
]

def select_columns(input_file_path, output_file_path, columns_to_keep):
    try:
        df = pd.read_csv(input_file_path)
        df_selected = df[columns_to_keep]
        df_selected.to_csv(output_file_path, index=False)
        return df_selected
        
    except Exception as e:
        print(f"{str(e)}")
        return None

if __name__ == "__main__":
    input_file = "Data/Original/airbnb_test.csv"  # CSV original
    output_file = "Data/Clean/test_partial.csv"  # CSV nettoyé
    
    df_cleaned = select_columns(input_file, output_file, selected_columns)