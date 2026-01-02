#!/usr/bin/env python3
"""
Classifie automatiquement les fichiers Markdown par catégorie en utilisant Ollama.

Ce script utilise un modèle LLM local (llama3) via Ollama pour analyser le contenu
de chaque note et suggérer une catégorie appropriée.

Usage:
    python classify_with_ollama.py ./mon_dossier
    python classify_with_ollama.py ./mon_dossier --include-subfolders  # Inclure sous-dossiers
    python classify_with_ollama.py ./mon_dossier --model llama3:8b-instruct-q4_0
    python classify_with_ollama.py ./mon_dossier --dry-run             # Simulation
"""

import argparse
import json
import locale
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple
import urllib.request
import urllib.error


# Configuration par défaut
SCRIPT_DIR = Path(__file__).parent
CONFIG_FILE = SCRIPT_DIR / "config.json"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
PROMPTS_DIR = SCRIPT_DIR / "prompts"

# Détection de la langue système
def get_system_language() -> str:
    """Détecte la langue du système (fr ou en)."""
    try:
        # Sur Windows, utiliser plusieurs méthodes
        if os.name == 'nt':  # Windows
            # Méthode 1: Utiliser PowerShell pour récupérer la culture
            try:
                result = subprocess.run(
                    ['powershell', '-Command', '(Get-Culture).Name'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    culture = result.stdout.strip()
                    if culture.startswith('fr'):
                        return "fr"
            except:
                pass
            
            # Méthode 2: Variables d'environnement Windows
            for var in ['LC_ALL', 'LC_CTYPE', 'LANG', 'LANGUAGE']:
                lang = os.environ.get(var, "")
                if lang.startswith('fr'):
                    return "fr"
        
        # Fallback : utiliser les variables d'environnement (Linux/macOS)
        lang = os.environ.get("LANG", "") or os.environ.get("LANGUAGE", "") or os.environ.get("LC_ALL", "")
        return "fr" if lang.startswith("fr") else "en"
    except:
        return "en"

LANG = get_system_language()

# Fichiers de configuration selon la langue
CATEGORIES_FILE = SCRIPT_DIR / ("categories.txt" if LANG == "fr" else "categories_en.txt")
# Fallback sur categories.txt si le fichier anglais n'existe pas
if not CATEGORIES_FILE.exists():
    CATEGORIES_FILE = SCRIPT_DIR / "categories.txt"

# Configuration par défaut
DEFAULT_CONFIG = {
    "source_dir": "",
    "model": "",
    "dry_run": False,
    "include_subfolders": False,
    "language": "",  # Vide = auto-détection
}

# Messages selon la langue
MESSAGES = {
    "fr": {
        "config_title": "📁 Configuration du classement",
        "source_dir": "Dossier source",
        "no_dir": "❌ Aucun dossier spécifié",
        "invalid_dir": "❌ Dossier invalide",
        "models_title": "🧠 Modèles Ollama disponibles:",
        "choice": "Choix",
        "classifying": "Classification de {count} fichiers avec {model}...",
        "categories_available": "Catégories disponibles:",
        "mode": "Mode:",
        "simulation": "SIMULATION (dry-run)",
        "real": "RÉEL",
        "summary": "RÉSUMÉ DE LA CLASSIFICATION",
        "files": "fichiers",
        "todo_found": "📝 {count} fichier(s) dans le dossier 'À faire'.",
        "todo_classified": "📝 {count} fichier(s) classé(s) 'À faire'.",
        "simulation_warning": "⚠️  Mode simulation - aucun fichier n'a été déplacé",
        "miscellaneous": "Divers",
    },
    "en": {
        "config_title": "📁 Classification setup",
        "source_dir": "Source folder",
        "no_dir": "❌ No folder specified",
        "invalid_dir": "❌ Invalid folder",
        "models_title": "🧠 Available Ollama models:",
        "choice": "Choice",
        "classifying": "Classifying {count} files with {model}...",
        "categories_available": "Available categories:",
        "mode": "Mode:",
        "simulation": "SIMULATION (dry-run)",
        "real": "REAL",
        "summary": "CLASSIFICATION SUMMARY",
        "files": "files",
        "todo_found": "📝 {count} file(s) in 'To do' folder.",
        "todo_classified": "📝 {count} file(s) classified as 'To do'.",
        "simulation_warning": "⚠️  Simulation mode - no files were moved",
        "miscellaneous": "Miscellaneous",
    }
}

def msg(key: str, **kwargs) -> str:
    """Retourne un message dans la langue courante."""
    text = MESSAGES.get(LANG, MESSAGES["en"]).get(key, key)
    if kwargs:
        text = text.format(**kwargs)
    return text

# Modèles llama3 selon la puissance du PC
LLAMA3_MODELS = {
    "high": "llama3:8b-instruct-q8_0",      # Haute qualité, ~8GB VRAM
    "medium": "llama3:8b-instruct-q4_0",    # Bon compromis, ~5GB VRAM
    "low": "llama3:8b-instruct-q4_0",       # Même modèle mais on limite le contexte
}

# Cache du modèle sélectionné
_selected_model = None

# Catégories par défaut (utilisées si categories.txt n'existe pas)
DEFAULT_CATEGORIES_FR = [
    "Courses", "Recettes", "Sport", "Informatique", "Travail",
    "Santé", "Finance", "Loisirs", "Personnel", "À faire",
]
DEFAULT_CATEGORIES_EN = [
    "Shopping", "Recipes", "Sports", "Tech", "Work",
    "Health", "Finance", "Entertainment", "Personal", "To do",
]
DEFAULT_CATEGORIES = DEFAULT_CATEGORIES_FR if LANG == "fr" else DEFAULT_CATEGORIES_EN


def load_config() -> dict:
    """Charge ou crée la configuration."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except:
            pass
    return DEFAULT_CONFIG.copy()


def save_config(config: dict) -> None:
    """Sauvegarde la configuration."""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_categories() -> Tuple[List[str], List[str]]:
    """Charge les catégories depuis categories.txt ou utilise les valeurs par défaut.
    
    Returns:
        Tuple (liste des catégories, liste des lignes complètes avec indices)
    """
    if not CATEGORIES_FILE.exists():
        print(f"   ⚠️  Fichier {CATEGORIES_FILE.name} non trouvé, utilisation des catégories par défaut")
        return DEFAULT_CATEGORIES, [f"{cat} # " for cat in DEFAULT_CATEGORIES]
    
    categories = []
    categories_with_hints = []
    try:
        with open(CATEGORIES_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Ignorer les lignes vides et les commentaires purs
                if not line or line.startswith('#'):
                    continue
                
                # Séparer la catégorie de l'indice
                if '#' in line:
                    parts = line.split('#', 1)  # Split seulement au premier #
                    cat_part = parts[0].strip()
                    hint_part = parts[1].strip() if len(parts) > 1 else ""
                    if cat_part:
                        categories.append(cat_part)
                        # Format propre : "Catégorie # Indice"
                        categories_with_hints.append(f"{cat_part} # {hint_part}")
                else:
                    categories.append(line)
                    categories_with_hints.append(line)
        
        if categories:
            return categories, categories_with_hints
    except Exception as e:
        print(f"   ⚠️  Erreur lecture {CATEGORIES_FILE.name}: {e}")
    
    return DEFAULT_CATEGORIES, [f"{cat} # " for cat in DEFAULT_CATEGORIES]


def load_prompt(name: str) -> Optional[str]:
    """Charge un prompt depuis le dossier prompts/ selon la langue."""
    # Essayer d'abord le fichier spécifique à la langue
    if LANG == "en":
        prompt_file = PROMPTS_DIR / f"{name}_en.txt"
        if prompt_file.exists():
            try:
                return prompt_file.read_text(encoding='utf-8')
            except Exception as e:
                print(f"   ⚠️  Erreur lecture prompt {name}_en: {e}")
    
    # Fallback sur le fichier par défaut (français)
    prompt_file = PROMPTS_DIR / f"{name}.txt"
    if prompt_file.exists():
        try:
            return prompt_file.read_text(encoding='utf-8')
        except Exception as e:
            print(f"   ⚠️  Erreur lecture prompt {name}: {e}")
    return None


def get_system_power_level() -> Tuple[str, float, float]:
    """Détecte la puissance du système et retourne un niveau (low, medium, high).
    
    Basé sur:
    - RAM disponible
    - Présence et VRAM GPU (si nvidia-smi disponible)
    """
    ram_gb = 0.0
    vram_gb = 0.0
    
    # Détecter la RAM système
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    ram_kb = int(line.split()[1])
                    ram_gb = ram_kb / (1024 * 1024)
                    break
    except:
        ram_gb = 8.0  # Valeur par défaut
    
    # Détecter la VRAM GPU (NVIDIA)
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            vram_mb = int(result.stdout.strip().split('\n')[0])
            vram_gb = vram_mb / 1024
    except:
        vram_gb = 0.0  # Pas de GPU NVIDIA
    
    # Déterminer le niveau de puissance
    if (ram_gb >= 16 and vram_gb >= 8) or ram_gb >= 32 or vram_gb >= 12:
        return 'high', ram_gb, vram_gb
    elif (ram_gb >= 8 and vram_gb >= 4) or ram_gb >= 16 or vram_gb >= 6:
        return 'medium', ram_gb, vram_gb
    else:
        return 'low', ram_gb, vram_gb


def get_available_models() -> List[str]:
    """Récupère la liste des modèles Ollama installés."""
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            return [line.split()[0] for line in lines if line.strip()]
    except:
        pass
    return []


def select_and_ensure_model() -> str:
    """Sélectionne le meilleur modèle llama3 selon la puissance du PC et l'installe si nécessaire."""
    global _selected_model
    
    if _selected_model:
        return _selected_model
    
    print("\n🤖 Détection de la configuration système...")
    
    power_level, ram_gb, vram_gb = get_system_power_level()
    power_icons = {'high': '🚀', 'medium': '💻', 'low': '📱'}
    power_names = {'high': 'Élevée', 'medium': 'Moyenne', 'low': 'Limitée'}
    
    print(f"   {power_icons[power_level]} Puissance: {power_names[power_level]} (RAM: {ram_gb:.1f}GB, VRAM: {vram_gb:.1f}GB)")
    
    # Modèle recommandé selon la puissance
    recommended_model = LLAMA3_MODELS[power_level]
    
    # Vérifier les modèles disponibles
    available = get_available_models()
    llama3_models = [m for m in available if m.startswith('llama3')]
    
    print(f"   📋 Modèles llama3 disponibles: {llama3_models if llama3_models else 'aucun'}")
    
    # Chercher le modèle recommandé ou un équivalent
    if recommended_model in available:
        _selected_model = recommended_model
        print(f"   ✅ Modèle sélectionné: {_selected_model}")
        return _selected_model
    
    # Chercher un modèle llama3 existant compatible
    for model in llama3_models:
        if 'instruct' in model or '8b' in model:
            _selected_model = model
            print(f"   ✅ Modèle alternatif trouvé: {_selected_model}")
            return _selected_model
    
    # Aucun modèle disponible, télécharger le recommandé
    print(f"   ⬇️  Téléchargement de {recommended_model}...")
    print(f"      (Cela peut prendre quelques minutes selon votre connexion)")
    
    try:
        result = subprocess.run(
            ['ollama', 'pull', recommended_model],
            check=True
        )
        _selected_model = recommended_model
        print(f"   ✅ {recommended_model} installé avec succès!")
        return _selected_model
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Échec du téléchargement: {e}")
        print(f"      Essayez manuellement: ollama pull {recommended_model}")
        sys.exit(1)
    except FileNotFoundError:
        print("   ❌ Ollama n'est pas installé!")
        print("      Installez Ollama: https://ollama.ai")
        sys.exit(1)


def get_default_model() -> str:
    """Retourne le modèle par défaut (après sélection automatique)."""
    global _selected_model
    if not _selected_model:
        _selected_model = select_and_ensure_model()
    return _selected_model


def slugify(value: str, max_length: int = 80) -> str:
    """Nettoie une chaîne pour un nom de dossier valide.
    
    Préserve les majuscules, accents et espaces.
    Supprime uniquement les caractères non autorisés dans les noms de fichiers.
    """
    value = value.strip()
    # Supprimer les caractères interdits dans les noms de fichiers/dossiers
    value = re.sub(r'[<>:"/\\|?*]', '', value)
    # Remplacer les espaces multiples par un seul espace
    value = re.sub(r'\s+', ' ', value)
    value = value.strip()
    if not value:
        value = "Divers"
    if len(value) > max_length:
        value = value[:max_length].rstrip()
    return value


def read_md_file(filepath: Path, max_chars: int = 1000) -> str:
    """Lit le contenu d'un fichier MD (limité pour l'API)."""
    try:
        content = filepath.read_text(encoding="utf-8")
        # Limiter la taille pour ne pas surcharger le LLM
        if len(content) > max_chars:
            content = content[:max_chars] + "\n[...tronqué...]"
        return content
    except Exception as e:
        print(f"  Erreur lecture {filepath}: {e}")
        return ""


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


def call_ollama(prompt: str, model: Optional[str] = None) -> Optional[str]:
    """Appelle l'API Ollama et retourne la réponse."""
    if model is None:
        model = get_default_model()
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # Réponses plus déterministes
            "num_predict": 50,   # Réponse courte attendue
        }
    }
    
    try:
        req = urllib.request.Request(
            OLLAMA_API_URL,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "").strip()
            
    except urllib.error.URLError as e:
        print(f"  Erreur connexion Ollama: {e}")
        return None
    except Exception as e:
        print(f"  Erreur API Ollama: {e}")
        return None


def classify_note(title: str, content: str, categories: List[str], categories_with_hints: List[str], model: Optional[str] = None) -> Optional[str]:
    """Utilise Ollama pour classifier une note dans une catégorie."""
    
    categories_list = ", ".join(categories)
    hints_formatted = "\n".join(f"- {hint}" for hint in categories_with_hints)
    
    # Charger le prompt externe ou utiliser le prompt par défaut
    prompt_template = load_prompt("classify")
    if prompt_template:
        prompt = prompt_template.format(
            categories=categories_list,
            categories_with_hints=hints_formatted,
            title=title,
            content=content
        )
    else:
        # Prompt par défaut si le fichier n'existe pas
        prompt = f"""Tu es un assistant qui classifie des notes personnelles.
Réponds UNIQUEMENT avec le nom de la catégorie la plus appropriée.

Catégories disponibles avec indices:
{hints_formatted}

Si aucune catégorie ne correspond bien, réponds "Divers".

Titre: {title}

Contenu:
{content}

Catégorie:"""

    response = call_ollama(prompt, model)
    
    if not response:
        return None
    
    # Nettoyer la réponse (garder lettres, accents, espaces)
    response = response.strip()
    response = re.sub(r"[^a-zA-ZàâäéèêëïîôùûüçÀÂÄÉÈÊËÏÎÔÙÛÜÇ\s-]", "", response)
    response = response.strip()
    
    # Vérifier si la réponse est une catégorie valide (comparaison insensible à la casse)
    response_lower = response.lower()
    for cat in categories:
        if cat.lower() == response_lower:
            return cat
    
    # Essayer de trouver une correspondance partielle
    for cat in categories:
        if cat.lower() in response_lower or response_lower in cat.lower():
            return cat
    
    return "Divers"


def get_all_md_files(base_dir: Path, include_subfolders: bool = True) -> List[Path]:
    """Récupère tous les fichiers MD à classifier."""
    md_files = []
    
    if include_subfolders:
        # Récupérer les fichiers de sans-titre en priorité
        sans_titre_dir = base_dir / "sans-titre"
        if sans_titre_dir.exists():
            md_files.extend(list(sans_titre_dir.glob("*.md")))
        
        # Puis les fichiers à la racine
        md_files.extend([f for f in base_dir.glob("*.md") if f.is_file()])
    else:
        md_files = list(base_dir.glob("*.md"))
    
    return md_files


def generate_todo_file(todo_files: List[Path], output_dir: Path, dry_run: bool = False) -> Optional[Path]:
    """Génère un fichier Todo.md consolidé à partir des fichiers classés 'À faire'.
    
    Args:
        todo_files: Liste des fichiers classés 'À faire'
        output_dir: Dossier de destination
        dry_run: Si True, affiche le contenu sans créer le fichier
    
    Returns:
        Le chemin du fichier créé, ou None en mode dry_run
    """
    if not todo_files:
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
            # Supprimer les lignes vides au début
            body_lines = body.strip().split("\n")
            
            # Ignorer la première ligne si elle répète le titre
            if body_lines and body_lines[0].strip().lower() == title.lower():
                body_lines = body_lines[1:]
            
            # Garder le texte tel quel
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
        print(content[:1000])
        if len(content) > 1000:
            print("[...tronqué pour l'aperçu...]")
        return None
    
    # Créer le fichier
    todo_path = output_dir / "Todo.md"
    
    # Gérer les conflits de nom
    if todo_path.exists():
        suffix = 1
        while todo_path.exists():
            todo_path = output_dir / f"Todo_{suffix}.md"
            suffix += 1
    
    todo_path.write_text(content, encoding="utf-8")
    print(f"\n✅ Fichier Todo.md créé : {todo_path}")
    
    return todo_path


def classify_and_organize(
    md_dir: Path,
    categories: List[str],
    categories_with_hints: List[str],
    model: Optional[str] = None,
    dry_run: bool = False,
    limit: Optional[int] = None,
    include_subfolders: bool = False
) -> None:
    """Classifie et organise les fichiers MD par catégorie."""
    
    if not md_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {md_dir}")
    
    # Sélectionner automatiquement le modèle si non spécifié
    if model is None:
        model = select_and_ensure_model()
    
    # Récupérer les fichiers à classifier
    md_files = get_all_md_files(md_dir, include_subfolders=include_subfolders)
    
    if not md_files:
        print("Aucun fichier MD à classifier.")
        return
    
    if limit:
        md_files = md_files[:limit]
    
    print(msg("classifying", count=len(md_files), model=model))
    print(f"{msg('categories_available')} {', '.join(categories)}")
    mode_text = msg("simulation") if dry_run else msg("real")
    print(f"{msg('mode')} {mode_text}")
    print("-" * 60)
    
    # Statistiques
    stats = {cat: 0 for cat in categories}
    stats[msg("miscellaneous")] = 0
    stats["erreur" if LANG == "fr" else "error"] = 0
    
    # Collecter les fichiers classés 'À faire' / 'To do'
    todo_files: List[Path] = []
    
    for i, md_file in enumerate(md_files, 1):
        title = md_file.stem.replace("-", " ").title()
        content = read_md_file(md_file)
        
        if not content:
            stats["erreur"] += 1
            continue
        
        print(f"[{i}/{len(md_files)}] {md_file.name}...", end=" ", flush=True)
        
        # Classifier avec Ollama
        category = classify_note(title, content, categories, categories_with_hints, model)
        
        if category is None:
            print("ERREUR API")
            stats["erreur"] += 1
            continue
        
        print(f"→ {category}")
        stats[category] = stats.get(category, 0) + 1
        
        # Collecter les fichiers 'À faire' / 'To do'
        if category.lower() in ["à faire", "a faire", "a-faire", "to do", "todo"]:
            todo_files.append(md_file)
        
        # Déplacer le fichier
        if not dry_run:
            dest_dir = md_dir / slugify(category)
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            dest_path = dest_dir / md_file.name
            
            # Gérer les conflits de nom
            if dest_path.exists() and dest_path != md_file:
                base = dest_path.stem
                suffix = 1
                while dest_path.exists():
                    dest_path = dest_dir / f"{base}_{suffix}.md"
                    suffix += 1
            
            if md_file.parent != dest_dir:
                shutil.move(str(md_file), str(dest_path))
        
        # Petite pause pour ne pas surcharger Ollama
        time.sleep(0.1)
    
    # Résumé
    print("\n" + "=" * 60)
    print(msg("summary"))
    print("=" * 60)
    
    for cat, count in sorted(stats.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {cat}: {count} {msg('files')}")
    
    # Générer le fichier Todo.md à partir de TOUS les fichiers du dossier "À faire" / "To do"
    todo_dir_name = "À faire" if LANG == "fr" else "To do"
    todo_dir = md_dir / todo_dir_name
    if todo_dir.exists():
        all_todo_files = list(todo_dir.glob("*.md"))
        # Exclure le fichier Todo.md lui-même
        all_todo_files = [f for f in all_todo_files if f.name.lower() not in ["todo.md"]]
        if all_todo_files:
            print(msg("todo_found", count=len(all_todo_files)))
            generate_todo_file(all_todo_files, md_dir, dry_run)
    elif todo_files:
        # En mode dry-run, le dossier n'existe pas encore, utiliser les fichiers collectés
        print(msg("todo_classified", count=len(todo_files)))
        generate_todo_file(todo_files, md_dir, dry_run)
    
    if dry_run:
        print(f"\n{msg('simulation_warning')}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classifie automatiquement les fichiers Markdown avec Ollama."
    )
    parser.add_argument(
        "md_dir",
        nargs="?",
        default=None,
        help="Répertoire contenant les fichiers Markdown (interactif si non spécifié)"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Modèle Ollama à utiliser (auto-détecté si non spécifié)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Simulation sans déplacer les fichiers"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limiter le nombre de fichiers à traiter"
    )
    parser.add_argument(
        "--include-subfolders", "-s",
        action="store_true",
        help="Inclure les fichiers des sous-dossiers (par défaut: racine uniquement)"
    )
    parser.add_argument(
        "--categories", "-c",
        nargs="+",
        default=None,
        help="Liste des catégories (sinon chargées depuis categories.txt)"
    )
    
    args = parser.parse_args(argv)
    
    # Charger la configuration
    config = load_config()
    
    # Mode interactif si aucun dossier spécifié
    if args.md_dir is None:
        print(f"\n{msg('config_title')}")
        print("-" * 40)
        
        # Choix du dossier
        default_dir = config.get("source_dir", "")
        prompt_dir = f"{msg('source_dir')} [{default_dir}]: " if default_dir else f"{msg('source_dir')}: "
        choix_dir = input(prompt_dir).strip()
        
        if choix_dir:
            md_dir = Path(choix_dir).expanduser().resolve()
        elif default_dir:
            md_dir = Path(default_dir).expanduser().resolve()
        else:
            print(msg("no_dir"))
            return 1
        
        if not md_dir.is_dir():
            print(f"{msg('invalid_dir')}: {md_dir}")
            return 1
        
        config["source_dir"] = str(md_dir)
        
        # Choix du modèle
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]
                models = [line.split()[0] for line in lines if line.strip()]
                
                if models:
                    print(f"\n{msg('models_title')}")
                    for i, m in enumerate(models, 1):
                        print(f"   {i}. {m}")
                    
                    default_model = config.get("model", "")
                    prompt_model = f"{msg('choice')} [{default_model}]: " if default_model else f"{msg('choice')}: "
                    choix_model = input(prompt_model).strip()
                    
                    if choix_model.isdigit() and 1 <= int(choix_model) <= len(models):
                        args.model = models[int(choix_model) - 1]
                    elif choix_model:
                        args.model = choix_model
                    elif default_model:
                        args.model = default_model
                    
                    if args.model:
                        config["model"] = args.model
        except:
            pass
        
        # Sauvegarder la configuration
        save_config(config)
        print()
    else:
        md_dir = Path(args.md_dir).expanduser().resolve()
    
    # Charger les catégories depuis le fichier ou les arguments
    if args.categories:
        categories = args.categories
        categories_with_hints = args.categories  # Pas d'indices si passé en argument
    else:
        categories, categories_with_hints = load_categories()
    
    try:
        classify_and_organize(
            md_dir=md_dir,
            categories=categories,
            categories_with_hints=categories_with_hints,
            model=args.model,
            dry_run=args.dry_run,
            limit=args.limit,
            include_subfolders=args.include_subfolders
        )
    except Exception as e:
        print(f"Erreur: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
