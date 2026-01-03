#!/usr/bin/env python3
"""
Script de test pour la fonction extract_attachments_from_md
"""

import sys
import os

# Ajouter le dossier parent au chemin pour importer le module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classify_with_ollama import extract_attachments_from_md

def test_attachment_extraction():
    """Teste la fonction d'extraction des pièces jointes."""
    
    # Test 1: Contenu avec pièces jointes au format ![alt](attachment://filename)
    content1 = """
# Ma note

Voici une image: ![Photo de famille](attachment://family.jpg)

Et un PDF: ![Document important](attachment://contract.pdf)

Et un document Word: ![Rapport](attachment://report.docx)
"""
    
    print("Test 1: Pièces jointes avec format ![alt](attachment://filename)")
    attachments1 = extract_attachments_from_md(content1)
    print(f"Nombre de pièces jointes trouvées: {len(attachments1)}")
    for att in attachments1:
        print(f"  - {att['filename']} ({att['type']})")
    print()
    
    # Test 2: Contenu avec pièces jointes au format direct [attachment://filename]
    content2 = """
# Ma note

Voici une image: [attachment://photo.jpg]

Et un fichier Excel: [attachment://budget.xlsx]
"""
    
    print("Test 2: Pièces jointes avec format [attachment://filename]")
    attachments2 = extract_attachments_from_md(content2)
    print(f"Nombre de pièces jointes trouvées: {len(attachments2)}")
    for att in attachments2:
        print(f"  - {att['filename']} ({att['type']})")
    print()
    
    # Test 3: Contenu sans pièces jointes
    content3 = """
# Ma note

Ceci est une note sans pièces jointes.
"""
    
    print("Test 3: Contenu sans pièces jointes")
    attachments3 = extract_attachments_from_md(content3)
    print(f"Nombre de pièces jointes trouvées: {len(attachments3)}")
    print()
    
    # Test 4: Contenu avec différents types de fichiers
    content4 = """
# Ma note

Image: ![Photo](attachment://image.png)
PDF: ![Document](attachment://doc.pdf)
Word: ![Rapport](attachment://rapport.docx)
Excel: ![Budget](attachment://budget.xlsx)
PowerPoint: ![Présentation](attachment://presentation.pptx)
Audio: ![Enregistrement](attachment://audio.mp3)
Video: ![Vidéo](attachment://video.mp4)
Archive: ![Backup](attachment://backup.zip)
Text: ![Notes](attachment://notes.txt)
"""
    
    print("Test 4: Différents types de fichiers")
    attachments4 = extract_attachments_from_md(content4)
    print(f"Nombre de pièces jointes trouvées: {len(attachments4)}")
    for att in attachments4:
        print(f"  - {att['filename']} ({att['type']})")
    print()
    
    # Vérifier que les types sont correctement détectés
    expected_types = {
        'image.png': 'image',
        'doc.pdf': 'pdf',
        'rapport.docx': 'word',
        'budget.xlsx': 'excel',
        'presentation.pptx': 'powerpoint',
        'audio.mp3': 'audio',
        'video.mp4': 'video',
        'backup.zip': 'archive',
        'notes.txt': 'text'
    }
    
    print("Vérification des types détectés:")
    all_correct = True
    for att in attachments4:
        expected_type = expected_types.get(att['filename'])
        if expected_type and att['type'] != expected_type:
            print(f"  ❌ {att['filename']}: attendu {expected_type}, obtenu {att['type']}")
            all_correct = False
        else:
            print(f"  ✅ {att['filename']}: {att['type']}")
    
    if all_correct:
        print("\n✅ Tous les types de fichiers ont été correctement détectés!")
    else:
        print("\n❌ Certains types de fichiers n'ont pas été correctement détectés.")

if __name__ == "__main__":
    test_attachment_extraction()