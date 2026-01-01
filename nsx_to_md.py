#!/usr/bin/env python3
"""
Convert Synology Note Station .nsx exports to Markdown files.

Format attendu :
- config.json à la racine, avec :
    {
      "note": ["ID1", "ID2", ...],
      "notebook": [...],
      "shortcut": {...}
    }
- un fichier JSON par note à la racine :
    ID1.json, ID2.json, ...

Usage:
    python nsx_to_md.py export.nsx ./export_md
"""

import argparse
import html
import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# Fichiers de notes à la racine : juste l'ID sans extension
NOTE_PATTERN = "{id}"


def html_to_markdown(html_content: str) -> str:
    """Convertit le contenu HTML en Markdown basique."""
    if not html_content:
        return ""
    
    text = html_content
    
    # Remplacer les balises de titre
    text = re.sub(r'<h1[^>]*>(.*?)</h1>', r'# \1\n', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<h2[^>]*>(.*?)</h2>', r'## \1\n', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<h3[^>]*>(.*?)</h3>', r'### \1\n', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<h4[^>]*>(.*?)</h4>', r'#### \1\n', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remplacer les balises de formatage
    text = re.sub(r'<strong[^>]*>(.*?)</strong>', r'**\1**', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<b[^>]*>(.*?)</b>', r'**\1**', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<em[^>]*>(.*?)</em>', r'*\1*', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<i[^>]*>(.*?)</i>', r'*\1*', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<u[^>]*>(.*?)</u>', r'\1', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<s[^>]*>(.*?)</s>', r'~~\1~~', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<strike[^>]*>(.*?)</strike>', r'~~\1~~', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remplacer les liens
    text = re.sub(r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>', r'[\2](\1)', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remplacer les listes
    text = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1\n', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'</?[uo]l[^>]*>', '', text, flags=re.IGNORECASE)
    
    # Remplacer les sauts de ligne
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<div[^>]*>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'</div>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<p[^>]*>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE)
    
    # Supprimer les autres balises HTML
    text = re.sub(r'<[^>]+>', '', text)
    
    # Décoder les entités HTML
    text = html.unescape(text)
    
    # Nettoyer les espaces multiples et lignes vides
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()


def format_timestamp(ts: Any) -> str:
    """Convertit un timestamp Unix en date lisible."""
    if ts is None:
        return ""
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        return str(ts)
    except (ValueError, OSError):
        return str(ts)


def slugify(value: str, max_length: int = 80) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    if not value:
        value = "note"
    if len(value) > max_length:
        value = value[:max_length].rstrip("-")
    return value


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_config(zf: zipfile.ZipFile) -> Dict[str, Any]:
    if "config.json" not in zf.namelist():
        raise FileNotFoundError("config.json not found in NSX archive.")
    with zf.open("config.json") as f:
        return json.load(f)


def load_note_json(zf: zipfile.ZipFile, note_id: str) -> Dict[str, Any]:
    """
    Charge le JSON d'une note à partir de son ID.

    Ici, chaque note est stockée directement avec son ID comme nom de fichier (sans extension).
    """
    candidate = NOTE_PATTERN.format(id=note_id)
    names = zf.namelist()

    if candidate not in names:
        # Fallback : chercher un fichier qui contient l'ID
        for name in names:
            if note_id in name:
                candidate = name
                break
        else:
            raise FileNotFoundError(f"Note file '{note_id}' not found in NSX archive.")

    with zf.open(candidate) as f:
        return json.load(f)


def note_to_markdown(title: str, content: str, meta: Dict[str, Any]) -> str:
    lines: List[str] = []

    if title:
        lines.append(f"# {title}")
        lines.append("")

    # Récupérer les timestamps (format NSX: ctime/mtime en secondes Unix)
    created = (
        meta.get("ctime")
        or meta.get("created")
        or meta.get("createTime")
        or meta.get("ct")
    )
    updated = (
        meta.get("mtime")
        or meta.get("updated")
        or meta.get("updateTime")
        or meta.get("ut")
    )
    tags = meta.get("tag") or meta.get("tags") or meta.get("labels")

    meta_lines = []
    if created:
        meta_lines.append(f"created: {format_timestamp(created)}")
    if updated:
        meta_lines.append(f"updated: {format_timestamp(updated)}")
    if tags:
        if isinstance(tags, list):
            tags_str = ", ".join(map(str, tags))
        else:
            tags_str = str(tags)
        meta_lines.append(f"tags: {tags_str}")

    if meta_lines:
        lines.append("<!--")
        lines.extend(meta_lines)
        lines.append("-->")
        lines.append("")

    if content:
        lines.append(content)

    return "\n".join(lines).rstrip() + "\n"


def convert_nsx_to_markdown(nsx_path: Path, output_dir: Path) -> None:
    if not nsx_path.is_file():
        raise FileNotFoundError(f"NSX file not found: {nsx_path}")

    ensure_dir(output_dir)

    print(f"Opening {nsx_path} ...")
    with zipfile.ZipFile(nsx_path, "r") as zf:
        config = load_config(zf)

        note_ids = config.get("note")
        if not isinstance(note_ids, list):
            raise ValueError("config.json does not contain a 'note' array with note IDs.")

        converted = 0

        for note_id in note_ids:
            if not isinstance(note_id, str):
                continue

            note_obj = load_note_json(zf, note_id)

            title = (
                note_obj.get("subject")
                or note_obj.get("title")
                or note_obj.get("name")
                or "Untitled"
            )
            title = str(title)

            # Contenu : le contenu est en HTML, on le convertit en Markdown
            html_content = (
                note_obj.get("content")
                or note_obj.get("body")
                or note_obj.get("text")
                or ""
            )
            if not isinstance(html_content, str):
                html_content = str(html_content)
            
            # Convertir le HTML en Markdown
            content = html_to_markdown(html_content)

            md_text = note_to_markdown(title, content, note_obj)

            filename = f"{slugify(title)}.md"
            out_path = output_dir / filename
            out_path.write_text(md_text, encoding="utf-8")

            converted += 1
            print(f"Converted: {title!r} ({note_id}) → {out_path}")

    print(f"Done. Converted {converted} notes into {output_dir}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Synology Note Station .nsx exports to Markdown files."
    )
    parser.add_argument("nsx_file", help="Path to the .nsx export file")
    parser.add_argument("output_dir", help="Directory where Markdown files will be written")
    args = parser.parse_args(argv)

    nsx_path = Path(args.nsx_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    try:
        convert_nsx_to_markdown(nsx_path, output_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
