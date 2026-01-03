#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test avec le fichier qui a un attachement"""

import sys
import os
from pathlib import Path

# Forcer l'encodage UTF-8
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.insert(0, r"C:\Users\guena\SynologyDrive\Drive\Développement\Classement notes")

from classify_with_ollama import extract_attachments, move_attachments, read_md_file

# Test sur le fichier test_attachement.md
test_file = Path(r"C:\Users\guena\SynologyDrive\Drive\Développement\Classement notes Fichiers tests\test_attachement.md")
dest_dir = Path(r"C:\Users\guena\SynologyDrive\Drive\Développement\Classement notes Fichiers tests\Recettes")

print(f"Test du fichier: {test_file.name}")
print(f"Fichier existe: {test_file.exists()}")

if test_file.exists():
    content = test_file.read_text(encoding="utf-8")
    print(f"\nContenu (100 premiers caractères):\n{content[:100]}")
    
    # Test extraction des attachements
    attachments = extract_attachments(content, test_file)
    print(f"\nAttachements trouvés: {len(attachments)}")
    for att in attachments:
        print(f"  - {att}")
        print(f"    Existe: {att.exists()}")
    
    # Test déplacement en mode simulation
    if attachments:
        print(f"\n--- Test déplacement (simulation) ---")
        moved = move_attachments(test_file, dest_dir, dry_run=True)
        print(f"Fichiers qui seraient déplacés: {moved}")
