import pandas as pd
import os

def remove_coordinates_columns(file_path):
    """
    Supprime les colonnes longitude et latitude du fichier CSV
    et sauvegarde sous le même nom
    """
    print(f"📂 Chargement du fichier: {file_path}")
    
    try:
        # Charger le fichier
        df = pd.read_csv(file_path)
        print(f"✅ Fichier chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        
        # Afficher les colonnes actuelles
        print(f"\n📋 Colonnes actuelles:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # Identifier les colonnes à supprimer
        columns_to_remove = []
        for col in df.columns:
            if 'longitude' in col.lower() or 'latitude' in col.lower():
                columns_to_remove.append(col)
        
        if columns_to_remove:
            print(f"\n🗑️  Colonnes à supprimer détectées:")
            for col in columns_to_remove:
                print(f"   - {col}")
            
            # Supprimer les colonnes
            df_cleaned = df.drop(columns=columns_to_remove)
            
            print(f"\n✂️  Colonnes supprimées: {len(columns_to_remove)}")
            print(f"📊 Nouveau dataset: {df_cleaned.shape[0]} lignes, {df_cleaned.shape[1]} colonnes")
            
            # Créer une sauvegarde avant modification
            backup_path = file_path.replace('.csv', '_backup.csv')
            df.to_csv(backup_path, index=False)
            print(f"💾 Sauvegarde créée: {backup_path}")
            
            # Sauvegarder le fichier nettoyé sous le même nom
            df_cleaned.to_csv(file_path, index=False)
            print(f"💾 Fichier mis à jour: {file_path}")
            
            print(f"\n📋 Colonnes finales:")
            for i, col in enumerate(df_cleaned.columns, 1):
                print(f"  {i:2d}. {col}")
                
        else:
            print(f"\n⚠️  Aucune colonne longitude/latitude trouvée dans le fichier")
            print(f"    Colonnes recherchées: 'longitude', 'latitude' (insensible à la casse)")
        
        print(f"\n🎉 TRAITEMENT TERMINÉ!")
        
    except FileNotFoundError:
        print(f"❌ Erreur: Fichier {file_path} non trouvé!")
        
    except Exception as e:
        print(f"❌ Erreur lors du traitement: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Fonction principale"""
    file_path = "Data/Clean/test_final_model.csv"
    
    # Vérifier l'existence du fichier
    if not os.path.exists(file_path):
        print(f"❌ Le fichier {file_path} n'existe pas!")
        print("💡 Fichiers disponibles dans le dossier Data/Clean/:")
        
        clean_dir = "Data/Clean"
        if os.path.exists(clean_dir):
            csv_files = [f for f in os.listdir(clean_dir) if f.endswith('.csv')]
            if csv_files:
                for f in csv_files:
                    print(f"   - {f}")
            else:
                print("   Aucun fichier CSV trouvé")
        else:
            print(f"   Le dossier {clean_dir} n'existe pas")
        return
    
    # Supprimer les coordonnées
    remove_coordinates_columns(file_path)

if __name__ == "__main__":
    main()