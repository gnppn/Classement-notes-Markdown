#!/usr/bin/env python3
"""
Organise les fichiers Markdown exportés par catégorie (notebook).

Ce script lit le fichier NSX original pour extraire les informations
de catégorisation et déplace les fichiers MD dans des sous-dossiers
correspondant à leurs notebooks.

Usage:
    python organize_md_by_category.py export.nsx ./export_md
"""

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def slugify(value: str, max_length: int = 80) -> str:
    """Convertit une chaîne en slug pour nom de fichier/dossier."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9àâäéèêëïîôùûüç]+", "-", value)
    value = value.strip("-")
    if not value:
        value = "sans-titre"
    if len(value) > max_length:
        value = value[:max_length].rstrip("-")
    return value


def ensure_dir(path: Path) -> None:
    """Crée un répertoire s'il n'existe pas."""
    path.mkdir(parents=True, exist_ok=True)


def load_config(zf: zipfile.ZipFile) -> Dict[str, Any]:
    """Charge le fichier config.json de l'archive NSX."""
    if "config.json" not in zf.namelist():
        raise FileNotFoundError("config.json not found in NSX archive.")
    with zf.open("config.json") as f:
        return json.load(f)


def load_item_json(zf: zipfile.ZipFile, item_id: str) -> Dict[str, Any]:
    """Charge le JSON d'un item (note ou notebook) à partir de son ID."""
    names = zf.namelist()
    if item_id not in names:
        raise FileNotFoundError(f"Item '{item_id}' not found in NSX archive.")
    with zf.open(item_id) as f:
        return json.load(f)


def build_notebook_map(zf: zipfile.ZipFile, config: Dict[str, Any]) -> Dict[str, str]:
    """
    Construit un dictionnaire {notebook_id: notebook_title}.
    """
    notebook_ids = config.get("notebook", [])
    notebook_map = {}
    
    for nb_id in notebook_ids:
        if not isinstance(nb_id, str):
            continue
        try:
            nb_data = load_item_json(zf, nb_id)
            title = nb_data.get("title") or nb_data.get("name") or "Sans titre"
            notebook_map[nb_id] = str(title)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load notebook {nb_id}: {e}")
            continue
    
    return notebook_map


def build_note_to_notebook_map(
    zf: zipfile.ZipFile, 
    config: Dict[str, Any]
) -> Dict[str, str]:
    """
    Construit un dictionnaire {note_id: parent_notebook_id}.
    """
    note_ids = config.get("note", [])
    note_to_notebook = {}
    
    for note_id in note_ids:
        if not isinstance(note_id, str):
            continue
        try:
            note_data = load_item_json(zf, note_id)
            parent_id = note_data.get("parent_id")
            if parent_id:
                note_to_notebook[note_id] = parent_id
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    
    return note_to_notebook


def get_note_title(zf: zipfile.ZipFile, note_id: str) -> str:
    """Récupère le titre d'une note."""
    try:
        note_data = load_item_json(zf, note_id)
        title = (
            note_data.get("title") 
            or note_data.get("subject") 
            or note_data.get("name") 
            or "Untitled"
        )
        return str(title)
    except (FileNotFoundError, json.JSONDecodeError):
        return "Untitled"


def find_md_file(md_dir: Path, title: str) -> Optional[Path]:
    """
    Trouve le fichier MD correspondant à un titre.
    Retourne le chemin du fichier ou None s'il n'existe pas.
    """
    expected_filename = f"{slugify(title)}.md"
    expected_path = md_dir / expected_filename
    
    if expected_path.exists():
        return expected_path
    
    # Fallback: chercher un fichier avec un nom similaire
    slug = slugify(title)
    for md_file in md_dir.glob("*.md"):
        if md_file.stem == slug or md_file.stem.startswith(slug[:20]):
            return md_file
    
    return None


def organize_by_category(nsx_path: Path, md_dir: Path) -> None:
    """
    Organise les fichiers MD par catégorie (notebook).
    """
    if not nsx_path.is_file():
        raise FileNotFoundError(f"NSX file not found: {nsx_path}")
    
    if not md_dir.is_dir():
        raise FileNotFoundError(f"Markdown directory not found: {md_dir}")
    
    print(f"Reading {nsx_path} ...")
    
    with zipfile.ZipFile(nsx_path, "r") as zf:
        config = load_config(zf)
        
        # Construire les mappings
        print("Building notebook map...")
        notebook_map = build_notebook_map(zf, config)
        print(f"Found {len(notebook_map)} notebooks:")
        for nb_id, nb_title in notebook_map.items():
            print(f"  - {nb_title}")
        
        print("\nBuilding note-to-notebook map...")
        note_to_notebook = build_note_to_notebook_map(zf, config)
        print(f"Found {len(note_to_notebook)} notes with parent notebooks")
        
        # Créer les dossiers de catégories
        category_dirs = {}
        for nb_id, nb_title in notebook_map.items():
            folder_name = slugify(nb_title)
            folder_path = md_dir / folder_name
            ensure_dir(folder_path)
            category_dirs[nb_id] = folder_path
        
        # Créer un dossier pour les notes sans catégorie
        uncategorized_dir = md_dir / "_sans-categorie"
        ensure_dir(uncategorized_dir)
        
        # Déplacer les fichiers MD
        moved = 0
        not_found = 0
        uncategorized = 0
        
        note_ids = config.get("note", [])
        
        for note_id in note_ids:
            if not isinstance(note_id, str):
                continue
            
            title = get_note_title(zf, note_id)
            md_file = find_md_file(md_dir, title)
            
            if md_file is None:
                not_found += 1
                continue
            
            # Déterminer le dossier destination
            parent_id = note_to_notebook.get(note_id)
            
            if parent_id and parent_id in category_dirs:
                dest_dir = category_dirs[parent_id]
                category_name = notebook_map[parent_id]
            else:
                dest_dir = uncategorized_dir
                category_name = "_sans-categorie"
                uncategorized += 1
            
            dest_path = dest_dir / md_file.name
            
            # Gérer les conflits de nom
            if dest_path.exists() and dest_path != md_file:
                base = dest_path.stem
                suffix = 1
                while dest_path.exists():
                    dest_path = dest_dir / f"{base}_{suffix}.md"
                    suffix += 1
            
            # Déplacer le fichier
            if md_file.parent != dest_dir:
                shutil.move(str(md_file), str(dest_path))
                moved += 1
                print(f"Moved: {md_file.name} → {category_name}/")
    
    print(f"\n--- Summary ---")
    print(f"Moved: {moved} files")
    print(f"Uncategorized: {uncategorized} files")
    print(f"Not found: {not_found} files")
    print(f"\nCategories created in {md_dir}:")
    for folder in sorted(md_dir.iterdir()):
        if folder.is_dir():
            count = len(list(folder.glob("*.md")))
            print(f"  {folder.name}: {count} files")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Organize Markdown files by category (notebook) from NSX export."
    )
    parser.add_argument("nsx_file", help="Path to the .nsx export file")
    parser.add_argument("md_dir", help="Directory containing the Markdown files")
    args = parser.parse_args(argv)
    
    nsx_path = Path(args.nsx_file).expanduser().resolve()
    md_dir = Path(args.md_dir).expanduser().resolve()
    
    try:
        organize_by_category(nsx_path, md_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
