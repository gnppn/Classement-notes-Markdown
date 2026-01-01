# Markdown Notes Classifier

⚠️ A small holiday **vibecoded** project. Don't expect anything reliable.

Inspired by [Note-Station-to-markdown](https://github.com/Maboroshy/Note-Station-to-markdown/). Run after to re-classify the exported notes.

## Features

- Automatic Markdown notes classification using AI (Ollama)
- Automatic PC power detection to choose the appropriate model
- Generation of a consolidated `Todo.md` file for tasks
- Multilingual interface (FR/EN based on system language)
- Interactive mode with configuration saving

## Quick Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Requirements

- Python 3.8+
- [Ollama](https://ollama.ai) installed (for AI classification)
- A llama3 model will be automatically downloaded if missing

## Usage

### Interactive Mode (recommended)

```bash
python classify_with_ollama.py
```

The script prompts for the source folder and model, then saves the configuration for future runs.

### Command Line Mode

```bash
# Classify Markdown files in a folder
python classify_with_ollama.py ./my_folder

# Include files from subfolders
python classify_with_ollama.py ./my_folder --include-subfolders

# Dry run without moving files
python classify_with_ollama.py ./my_folder --dry-run

# Limit the number of processed files
python classify_with_ollama.py ./my_folder --limit 50

# Force a specific Ollama model
python classify_with_ollama.py ./my_folder --model llama3:8b-instruct-q4_0
```

### Generate Todo.md Independently

```bash
# Generate a Todo.md from a folder
python generate_todo.py "./To do"

# Dry run mode
python generate_todo.py "./To do" --dry-run
```

## Configuration

### Categories (`categories.txt` / `categories_en.txt`)

One category per line with a hint for the AI after `#`.

```
Shopping        # Ingredient list WITHOUT instructions, just quantities
Recipes         # Cooking instructions WITH preparation steps
Sport           # Exercises, workouts, running
Personal        # Personal diary, food tracking with times
To do           # Tasks, todo lists
```

### AI Prompt (`prompts/classify.txt`)

Customize the prompt sent to the AI. Available variables:
- `{categories}`: list of categories
- `{categories_with_hints}`: categories with their hints
- `{title}`: note title
- `{content}`: note content

### Configuration (`config.json`)

Automatically created, saves:
- `source_dir`: last used folder
- `model`: last selected model

## Structure

```
Markdown Notes Classifier/
├── classify_with_ollama.py     # AI Classification
├── generate_todo.py            # Todo.md Generation
├── categories.txt              # FR Categories (editable)
├── categories_en.txt           # EN Categories (editable)
├── config.json                 # Configuration (auto-generated)
├── prompts/
│   ├── classify.txt            # FR Prompt
│   └── classify_en.txt         # EN Prompt
└── requirements.txt
```

## Reliability Notes

- Classification quality depends on the Ollama model and note content
- Very short or empty notes are difficult to classify
- Hints in `categories.txt` significantly improve accuracy

## License

MIT
