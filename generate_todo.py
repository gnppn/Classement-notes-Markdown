#!/usr/bin/env python3
"""
Génère un fichier Todo.md consolidé à partir des fichiers du dossier "À faire".

Usage:
    python generate_todo.py ./mon_dossier
    python generate_todo.py ./mon_dossier --output Todo.md
    python generate_todo.py  # Mode interactif
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Configuration
SCRIPT_DIR = Path(__file__).parent


def read_md_file_full(filepath: Path) -> Tuple[str, str, str]:
    """Lit le contenu complet d'un fichier MD et extrait le titre, la date et le contenu.
    
    Returns:
        Tuple (titre, date_modified, contenu_sans_frontmatter)
    """
    try:
        content = filepath.read_text(encoding="utf-8")
        title = filepath.stem
        modified = ""
        body = content
        
        # Extraire le frontmatter YAML si présent
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1]
                body = parts[2].strip()
                
                # Extraire Title et Modified du frontmatter
                for line in frontmatter.split("\n"):
                    if line.startswith("Title:"):
                        title = line.split(":", 1)[1].strip().strip('"')
                    elif line.startswith("Modified:"):
                        modified = line.split(":", 1)[1].strip().strip('"')
        
        return title, modified, body
    except Exception as e:
        print(f"  Erreur lecture {filepath}: {e}")
        return filepath.stem, "", ""


def generate_todo_file(todo_files: List[Path], output_path: Path, dry_run: bool = False) -> Optional[Path]:
    """Génère un fichier Todo.md consolidé.
    
    Args:
        todo_files: Liste des fichiers à inclure
        output_path: Chemin du fichier de sortie
        dry_run: Si True, affiche le contenu sans créer le fichier
    
    Returns:
        Le chemin du fichier créé, ou None en mode dry_run
    """
    if not todo_files:
        print("Aucun fichier à traiter.")
        return None
    
    # Construire le contenu du fichier Todo.md
    lines = [
        "# 📝 Liste des tâches",
        "",
        f"*Généré automatiquement le {time.strftime('%d/%m/%Y à %H:%M')}*",
        "",
        "---",
        ""
    ]
    
    for md_file in todo_files:
        title, modified, body = read_md_file_full(md_file)
        
        # Ajouter le titre de la tâche
        lines.append(f"## {title}")
        
        if modified:
            lines.append(f"*Modifié le {modified}*")
        
        lines.append("")
        
        # Nettoyer et formater le contenu
        if body:
            body_lines = body.strip().split("\n")
            
            # Ignorer la première ligne si elle répète le titre
            if body_lines and body_lines[0].strip().lower() == title.lower():
                body_lines = body_lines[1:]
            
            for line in body_lines:
                line = line.strip()
                if line:
                    lines.append(line)
            
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    content = "\n".join(lines)
    
    if dry_run:
        print("\n" + "=" * 60)
        print("📝 APERÇU DU FICHIER Todo.md")
        print("=" * 60)
        print(content[:2000])
        if len(content) > 2000:
            print("[...tronqué pour l'aperçu...]")
        return None
    
    # Créer le fichier
    output_path.write_text(content, encoding="utf-8")
    print(f"✅ Fichier créé : {output_path}")
    print(f"   {len(todo_files)} tâche(s) incluse(s)")
    
    return output_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Génère un fichier Todo.md à partir des fichiers d'un dossier."
    )
    parser.add_argument(
        "source_dir",
        nargs="?",
        default=None,
        help="Dossier contenant les fichiers (ex: ./À faire)"
    )
    parser.add_argument(
        "--output", "-o",
        default="Todo.md",
        help="Nom du fichier de sortie (défaut: Todo.md)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Afficher un aperçu sans créer le fichier"
    )
    
    args = parser.parse_args(argv)
    
    # Mode interactif si aucun dossier spécifié
    if args.source_dir is None:
        print("\n📝 Génération de Todo.md")
        print("-" * 40)
        source_dir = input("Dossier source (ex: ./À faire): ").strip()
        if not source_dir:
            print("❌ Aucun dossier spécifié")
            return 1
    else:
        source_dir = args.source_dir
    
    source_path = Path(source_dir).expanduser().resolve()
    
    if not source_path.is_dir():
        print(f"❌ Dossier invalide: {source_path}")
        return 1
    
    # Récupérer les fichiers .md
    md_files = list(source_path.glob("*.md"))
    # Exclure le fichier Todo.md lui-même
    md_files = [f for f in md_files if f.name.lower() != "todo.md"]
    
    if not md_files:
        print(f"Aucun fichier .md trouvé dans {source_path}")
        return 1
    
    print(f"\n📁 {len(md_files)} fichier(s) trouvé(s) dans {source_path.name}/")
    
    # Déterminer le chemin de sortie
    output_path = source_path.parent / args.output
    
    try:
        generate_todo_file(md_files, output_path, args.dry_run)
    except Exception as e:
        print(f"Erreur: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
