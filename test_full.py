#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test réel de déplacement avec attachement"""

import sys
import os
from pathlib import Path

# Forcer l'encodage UTF-8
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

os.chdir(r"C:\Users\guena\SynologyDrive\Drive\Développement\Classement notes Fichiers tests")
sys.path.insert(0, r"C:\Users\guena\SynologyDrive\Drive\Développement\Classement notes")

from classify_with_ollama import main

# Classifier uniquement le fichier test_attachement.md
print("=== Test de classification avec attachement ===\n")
sys.argv = ['classify_with_ollama.py', '.', '--limit', '10', '--dry-run']
sys.exit(main())
