import pandas as pd
import numpy as np

def remove_incomplete_listings(input_file_path, output_file_path, critical_columns=None):
    try:
        df = pd.read_csv(input_file_path)
        initial_shape = df.shape
        df = df.replace('', np.nan)
        df_cleaned = df.dropna()
        final_shape = df_cleaned.shape
        print(f"Lignes supprimées : {initial_shape[0] - final_shape[0]}")
        df_cleaned.to_csv(output_file_path, index=False)
        return df_cleaned
        
    except Exception as e:
        print(f"{str(e)}")
        return None

if __name__ == "__main__":
    input_file = "Data/Clean/test_partial.csv"  # CSV original
    output_file = "Data/Clean/test_partial_clean.csv"  # CSV nettoyé
    
    df_complete = remove_incomplete_listings(input_file, output_file)