import base64
import os
from bs4 import BeautifulSoup

# Installez BeautifulSoup si ce n'est pas déjà fait:
# pip install beautifulsoup4

def embed_images_in_html(html_file):
    print(f"Traitement du fichier: {html_file}")
    
    # Vérifier si le fichier existe
    if not os.path.exists(html_file):
        print(f"Le fichier {html_file} n'existe pas!")
        return
    
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
                # Construire le chemin absolu si nécessaire
                if os.path.isabs(src):
                    img_path = src
                else:
                    # Chemin relatif au fichier HTML
                    html_dir = os.path.dirname(os.path.abspath(html_file))
                    img_path = os.path.join(html_dir, src)
                
                print(f"[{i}/{len(img_tags)}] Traitement de l'image: {img_path}")
                
                # Vérifier si le fichier existe
                if os.path.exists(img_path):
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
                    print(f"  ✗ Image introuvable: {img_path}")
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
        # Demander le chemin du fichier HTML
        html_file = input("Entrez le chemin complet du fichier HTML exporté: ")
        embed_images_in_html(html_file)
    except Exception as e:
        print(f"Une erreur est survenue: {e}")
        input("Appuyez sur Entrée pour quitter...")