#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test du script classify_with_ollama.py avec gestion UTF-8"""

import sys
import os

# Forcer l'encodage UTF-8 pour la sortie
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Changer de répertoire
os.chdir(r"C:\Users\guena\SynologyDrive\Drive\Développement\Classement notes Fichiers tests")

# Exécuter le script
sys.path.insert(0, r"C:\Users\guena\SynologyDrive\Drive\Développement\Classement notes")

# Importer et exécuter
from classify_with_ollama import main

sys.argv = ['classify_with_ollama.py', '.', '--dry-run', '--limit', '2']
sys.exit(main())
