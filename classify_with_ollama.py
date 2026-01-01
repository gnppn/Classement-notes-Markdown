#!/usr/bin/env python3
"""
Classifie automatiquement les fichiers Markdown par catégorie en utilisant Ollama.

Ce script utilise un modèle LLM local (llama3) via Ollama pour analyser le contenu
de chaque note et suggérer une catégorie appropriée.

Usage:
    python classify_with_ollama.py ./export
    python classify_with_ollama.py ./export --model llama3:8b-instruct-q4_0
    python classify_with_ollama.py ./export --dry-run  # Simulation sans déplacement
"""

import argparse
import json
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
OLLAMA_API_URL = "http://localhost:11434/api/generate"
CATEGORIES_FILE = SCRIPT_DIR / "categories.txt"
PROMPTS_DIR = SCRIPT_DIR / "prompts"

# Modèles llama3 selon la puissance du PC
LLAMA3_MODELS = {
    "high": "llama3:8b-instruct-q8_0",      # Haute qualité, ~8GB VRAM
    "medium": "llama3:8b-instruct-q4_0",    # Bon compromis, ~5GB VRAM
    "low": "llama3:8b-instruct-q4_0",       # Même modèle mais on limite le contexte
}

# Cache du modèle sélectionné
_selected_model = None

# Catégories par défaut (utilisées si categories.txt n'existe pas)
DEFAULT_CATEGORIES = [
    "courses", "recettes", "sport", "informatique", "travail",
    "sante", "finance", "loisirs", "personnel", "a-faire",
]


def load_categories() -> List[str]:
    """Charge les catégories depuis categories.txt ou utilise les valeurs par défaut."""
    if not CATEGORIES_FILE.exists():
        print(f"   ⚠️  Fichier {CATEGORIES_FILE.name} non trouvé, utilisation des catégories par défaut")
        return DEFAULT_CATEGORIES
    
    categories = []
    try:
        with open(CATEGORIES_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.split('#')[0].strip()  # Ignorer les commentaires
                if line:
                    categories.append(line)
        if categories:
            return categories
    except Exception as e:
        print(f"   ⚠️  Erreur lecture {CATEGORIES_FILE.name}: {e}")
    
    return DEFAULT_CATEGORIES


def load_prompt(name: str) -> Optional[str]:
    """Charge un prompt depuis le dossier prompts/."""
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
    """Convertit une chaîne en slug pour nom de dossier."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9àâäéèêëïîôùûüç]+", "-", value)
    value = value.strip("-")
    if not value:
        value = "divers"
    if len(value) > max_length:
        value = value[:max_length].rstrip("-")
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


def classify_note(title: str, content: str, categories: List[str], model: Optional[str] = None) -> Optional[str]:
    """Utilise Ollama pour classifier une note dans une catégorie."""
    
    categories_list = ", ".join(categories)
    
    # Charger le prompt externe ou utiliser le prompt par défaut
    prompt_template = load_prompt("classify")
    if prompt_template:
        prompt = prompt_template.format(
            categories=categories_list,
            title=title,
            content=content
        )
    else:
        # Prompt par défaut si le fichier n'existe pas
        prompt = f"""Tu es un assistant qui classifie des notes personnelles.
Voici une note à classifier. Réponds UNIQUEMENT avec le nom de la catégorie la plus appropriée parmi: {categories_list}

Si aucune catégorie ne correspond bien, réponds "divers".

Titre: {title}

Contenu:
{content}

Catégorie:"""

    response = call_ollama(prompt, model)
    
    if not response:
        return None
    
    # Nettoyer la réponse
    response = response.lower().strip()
    response = re.sub(r"[^a-zàâäéèêëïîôùûüç-]", "", response)
    
    # Vérifier si la réponse est une catégorie valide
    if response in categories:
        return response
    
    # Essayer de trouver une correspondance partielle
    for cat in categories:
        if cat in response or response in cat:
            return cat
    
    return "divers"


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


def classify_and_organize(
    md_dir: Path,
    categories: List[str],
    model: Optional[str] = None,
    dry_run: bool = False,
    limit: Optional[int] = None
) -> None:
    """Classifie et organise les fichiers MD par catégorie."""
    
    if not md_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {md_dir}")
    
    # Sélectionner automatiquement le modèle si non spécifié
    if model is None:
        model = select_and_ensure_model()
    
    # Récupérer les fichiers à classifier
    md_files = get_all_md_files(md_dir)
    
    if not md_files:
        print("Aucun fichier MD à classifier.")
        return
    
    if limit:
        md_files = md_files[:limit]
    
    print(f"Classification de {len(md_files)} fichiers avec {model}...")
    print(f"Catégories disponibles: {', '.join(categories)}")
    print(f"Mode: {'SIMULATION (dry-run)' if dry_run else 'RÉEL'}")
    print("-" * 60)
    
    # Statistiques
    stats = {cat: 0 for cat in categories}
    stats["divers"] = 0
    stats["erreur"] = 0
    
    for i, md_file in enumerate(md_files, 1):
        title = md_file.stem.replace("-", " ").title()
        content = read_md_file(md_file)
        
        if not content:
            stats["erreur"] += 1
            continue
        
        print(f"[{i}/{len(md_files)}] {md_file.name}...", end=" ", flush=True)
        
        # Classifier avec Ollama
        category = classify_note(title, content, categories, model)
        
        if category is None:
            print("ERREUR API")
            stats["erreur"] += 1
            continue
        
        print(f"→ {category}")
        stats[category] = stats.get(category, 0) + 1
        
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
    print("RÉSUMÉ DE LA CLASSIFICATION")
    print("=" * 60)
    
    for cat, count in sorted(stats.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {cat}: {count} fichiers")
    
    if dry_run:
        print("\n⚠️  Mode simulation - aucun fichier n'a été déplacé")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classifie automatiquement les fichiers Markdown avec Ollama."
    )
    parser.add_argument(
        "md_dir", 
        help="Répertoire contenant les fichiers Markdown"
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
        "--categories", "-c",
        nargs="+",
        default=None,
        help="Liste des catégories (sinon chargées depuis categories.txt)"
    )
    
    args = parser.parse_args(argv)
    
    md_dir = Path(args.md_dir).expanduser().resolve()
    
    # Charger les catégories depuis le fichier ou les arguments
    if args.categories:
        categories = args.categories
    else:
        categories = load_categories()
    
    try:
        classify_and_organize(
            md_dir=md_dir,
            categories=categories,
            model=args.model,
            dry_run=args.dry_run,
            limit=args.limit
        )
    except Exception as e:
        print(f"Erreur: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
