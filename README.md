# Classement notes Markdown

⚠️ Petit projet de vacances **vibecodé**. N'en attendez rien de fiable.

Inspiré de [Note-Station-to-markdown](https://github.com/Maboroshy/Note-Station-to-markdown/). À lancer après pour re-classer les notes obtenues.

## Quoi

- Classification automatique de notes Markdown par IA (Ollama)
- Détection automatique de la puissance du PC pour choisir le bon modèle
- Génération d'un fichier `Todo.md` consolidé pour les tâches
- Interface multilingue (FR/EN selon la langue système)
- Mode interactif avec sauvegarde de la configuration

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

## Lancement

### Mode interactif (recommandé)

```bash
python classify_with_ollama.py
```

Le script demande le dossier source et le modèle, puis sauvegarde la configuration pour les prochaines exécutions.

### Mode ligne de commande

```bash
# Classifier les fichiers Markdown d'un dossier
python classify_with_ollama.py ./mon_dossier

# Inclure aussi les fichiers des sous-dossiers
python classify_with_ollama.py ./mon_dossier --include-subfolders

# Simulation sans déplacer les fichiers
python classify_with_ollama.py ./mon_dossier --dry-run

# Limiter le nombre de fichiers traités
python classify_with_ollama.py ./mon_dossier --limit 50

# Forcer un modèle Ollama spécifique
python classify_with_ollama.py ./mon_dossier --model llama3:8b-instruct-q4_0
```

### Générer Todo.md indépendamment

```bash
# Générer un Todo.md à partir d'un dossier
python generate_todo.py "./À faire"

# Mode simulation
python generate_todo.py "./À faire" --dry-run
```

## Configuration

### Catégories (`categories.txt` / `categories_en.txt`)

Une catégorie par ligne avec un indice pour l'IA après `#`.

```
Courses         # Liste d'ingrédients SANS instructions, juste des quantités
Recettes        # Instructions de cuisine AVEC étapes de préparation
Sport           # Exercices, entraînements, courses à pied
Personnel       # Journal intime, suivi alimentaire avec horaires
À faire         # Tâches, todo lists
```

### Prompt IA (`prompts/classify.txt`)

Personnalisez le prompt envoyé à l'IA. Variables disponibles :
- `{categories}` : liste des catégories
- `{categories_with_hints}` : catégories avec leurs indices
- `{title}` : titre de la note
- `{content}` : contenu de la note

### Configuration (`config.json`)

Créé automatiquement, sauvegarde :
- `source_dir` : dernier dossier utilisé
- `model` : dernier modèle sélectionné

## Structure

```
Classement notes Markdown/
├── classify_with_ollama.py     # Classification IA
├── generate_todo.py            # Génération Todo.md
├── categories.txt              # Catégories FR (éditable)
├── categories_en.txt           # Catégories EN (éditable)
├── config.json                 # Configuration (auto-généré)
├── prompts/
│   ├── classify.txt            # Prompt FR
│   └── classify_en.txt         # Prompt EN
└── requirements.txt
```

## Notes de fiabilité

- La qualité de la classification dépend du modèle Ollama et du contenu des notes
- Les notes très courtes ou vides sont difficiles à classifier
- Les indices dans `categories.txt` améliorent significativement la précision

## Licence

MIT
