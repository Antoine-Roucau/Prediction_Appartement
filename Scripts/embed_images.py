import base64
import os
from bs4 import BeautifulSoup
import sys

def embed_images_in_html(html_file, project_root=None):
    """
    Intègre les images en base64 dans un fichier HTML.
    
    Args:
        html_file: Chemin vers le fichier HTML à traiter
        project_root: Racine du projet pour résoudre les chemins relatifs (optionnel)
    """
    print(f"\n=== INTÉGRATION DES IMAGES DANS LE RAPPORT HTML ===")
    print(f"Traitement du fichier: {html_file}")
    
    # Vérifier si le fichier existe
    if not os.path.exists(html_file):
        print(f"Le fichier {html_file} n'existe pas!")
        return
    
    # Calculer la racine du projet si non fournie
    if project_root is None:
        # On remonte de 3 niveaux depuis le HTML (Data/Visual/picture)
        project_root = os.path.abspath(os.path.join(os.path.dirname(html_file), "../../.."))
    
    print(f"Racine du projet: {project_root}")
    
    # Lire le fichier HTML
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Analyser le HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Trouver toutes les balises img
    img_tags = soup.find_all('img')
    print(f"Trouvé {len(img_tags)} images à traiter")
    
    # Compteurs pour le rapport
    images_success = 0
    images_failed = 0
    
    # Remplacer chaque src par un data URI
    for i, img in enumerate(img_tags, 1):
        src = img.get('src')
        if src and not src.startswith('data:'):
            try:
                # Tenter différentes stratégies pour localiser l'image
                possible_paths = []
                
                # 1. Chemin tel quel (si absolu)
                if os.path.isabs(src):
                    possible_paths.append(src)
                    
                # 2. Chemin relatif au fichier HTML
                html_dir = os.path.dirname(os.path.abspath(html_file))
                possible_paths.append(os.path.join(html_dir, src))
                
                # 3. Chemin relatif à la racine du projet
                possible_paths.append(os.path.join(project_root, src))
                
                # 4. Essayer de corriger le chemin Windows vs Unix
                normalized_src = src.replace('/', '\\')
                possible_paths.append(os.path.join(project_root, normalized_src))
                
                # Essayer les différents chemins
                img_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        img_path = path
                        break
                        
                if img_path:
                    print(f"[{i}/{len(img_tags)}] Traitement de l'image: {img_path}")
                    
                    # Déterminer le type MIME
                    extension = os.path.splitext(img_path)[1].lower()
                    mime_types = {
                        '.png': 'image/png',
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.gif': 'image/gif',
                        '.svg': 'image/svg+xml'
                    }
                    mime_type = mime_types.get(extension, 'image/png')
                    
                    # Lire l'image et l'encoder en base64
                    with open(img_path, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                    
                    # Remplacer src par data URI
                    img['src'] = f"data:{mime_type};base64,{img_data}"
                    print(f"  ✓ Image intégrée avec succès: {os.path.basename(img_path)}")
                    images_success += 1
                else:
                    print(f"  ✗ Image introuvable: {src}")
                    print(f"    Chemins essayés: {possible_paths}")
                    images_failed += 1
            except Exception as e:
                print(f"  ✗ Erreur lors du traitement de {src}: {e}")
                images_failed += 1
    
    # Écrire le HTML modifié
    output_file = os.path.splitext(html_file)[0] + '_embedded.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(str(soup))
    
    print(f"\nRésumé:")
    print(f"- Images traitées avec succès: {images_success}")
    print(f"- Images qui ont échoué: {images_failed}")
    print(f"- Total: {len(img_tags)}")
    print(f"\nRésultat enregistré dans: {output_file}")
    
    return output_file

if __name__ == "__main__":
    try:
        # Chemin direct vers le rapport HTML
        html_file = "Data\\Visual\\picture\\rapportV3.html"
        
        # Si un argument est passé en ligne de commande, l'utiliser comme chemin du fichier HTML
        if len(sys.argv) > 1:
            html_file = sys.argv[1]
            
        print(f"Traitement du fichier: {html_file}")
        embed_images_in_html(html_file)
    except Exception as e:
        print(f"Une erreur est survenue: {e}")
        print("\nAppuyez sur Entrée pour quitter...")
        input()