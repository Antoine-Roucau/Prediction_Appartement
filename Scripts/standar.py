import pandas as pd

def standardize_property_types(input_file_path, output_file_path):
    try:
        df = pd.read_csv(input_file_path)
        property_types_to_keep = [
            'Apartment',
            'House',
            'Condominium',
            'Townhouse',
            'Loft',
            'Guesthouse'
        ]   
        df['property_type'] = df['property_type'].apply(
            lambda x: x if x in property_types_to_keep else 'Other'
        )
        df.to_csv(output_file_path, index=False)
        return df
        
    except Exception as e:
        print(f"={str(e)}")
        return None

if __name__ == "__main__":
    input_file = "Data/Clean/test_partial_clean.csv"  # CSV original
    output_file = "Data/Clean/test_partial_clean_standart.csv"  # CSV clean
    
    df_modified = standardize_property_types(input_file, output_file)