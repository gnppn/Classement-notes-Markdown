# nsx2md

⚠️ Petit projet de vacances **vibecodé**. N'en attendez rien de fiable.

Inspiré de [Note-Station-to-markdown](https://github.com/Maboroshy/Note-Station-to-markdown/).

## Quoi

- Un outil pour convertir les exports Synology Note Station (.nsx) en fichiers Markdown
- Conversion HTML → Markdown propre
- Organisation par notebooks d'origine
- Classification automatique par IA (Ollama) des notes non classées
- Détection automatique de la puissance du PC pour choisir le bon modèle

## Installation éclair

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Pré-requis rapides

- Python 3.8+
- [Ollama](https://ollama.ai) installé (pour la classification IA)
- Un modèle llama3 sera téléchargé automatiquement si absent

## Lancement minimal

```bash
# 1. Convertir le .nsx en fichiers Markdown
python nsx_to_md.py mon_export.nsx export

# 2. Organiser par notebook d'origine
python organize_md_by_category.py mon_export.nsx export

# 3. Classifier les notes "sans-titre" avec l'IA
python classify_with_ollama.py export
```

## Options utiles

```bash
# Simulation sans déplacer les fichiers
python classify_with_ollama.py export --dry-run

# Limiter le nombre de fichiers traités
python classify_with_ollama.py export --limit 50

# Forcer un modèle Ollama spécifique
python classify_with_ollama.py export --model llama3:8b-instruct-q4_0
```

## Configuration

### Catégories (`categories.txt`)

Une catégorie par ligne. Les commentaires `#` sont ignorés.

```
courses
recettes
sport
informatique
ma-nouvelle-categorie
```

### Prompt IA (`prompts/classify.txt`)

Personnalisez le prompt envoyé à l'IA. Variables : `{categories}`, `{title}`, `{content}`.

## Structure

```
nsx2md/
├── nsx_to_md.py                # Convertisseur NSX → Markdown
├── organize_md_by_category.py  # Classement par notebook d'origine
├── classify_with_ollama.py     # Classification IA
├── categories.txt              # Catégories (éditable)
├── prompts/classify.txt        # Prompt IA (éditable)
└── requirements.txt
```

## Notes de fiabilité

- Le format NSX n'est pas documenté, le script peut ne pas fonctionner avec tous les exports
- La qualité de la classification dépend du modèle Ollama et du contenu des notes
- Les notes très courtes ou vides sont difficiles à classifier

## État d'esprit

Ce dépôt est vraiment pensé en __one shot__ pour un besoin précis. Mais si ça peut servir...

## Licence

MIT
