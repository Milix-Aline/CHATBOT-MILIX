import os
from pathlib import Path
from dotenv import load_dotenv
import re

# Chemin du projet
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Créer les dossiers si nécessaire
ARTIFACTS_DIR.mkdir(exist_ok=True)

def find_env_file():
    """Trouve le fichier .env dans plusieurs emplacements possibles."""
    possible_paths = [
        PROJECT_ROOT / ".env",
        Path(__file__).parent / ".env",
        Path.cwd() / ".env"
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    return None

# Chargement des variables d'environnement
env_file = find_env_file()
if env_file:
    load_dotenv(env_file)
    print(f"Fichier .env chargé: {env_file}")
else:
    print("ATTENTION: Fichier .env non trouvé")

# Configuration Base de données
DB_TYPE = os.getenv("DB_TYPE", "sqlite")
DB_NAME = os.getenv("DB_NAME", "rag_system.db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "1234")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# Configuration Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama2:latest")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Configuration API
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = int(os.getenv("API_PORT", 8000))

# Chemins des modèles
CLF_PATH = ARTIFACTS_DIR / "emotion_clf.joblib"
LABELS_PATH = ARTIFACTS_DIR / "emotion_labels.json"

# Configuration modèle d'embeddings
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "all-MiniLM-L6-v2")

# Paramètres système
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 120))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
FORCE_FRENCH = os.getenv("FORCE_FRENCH", "false").lower() == "true"

# Multilingue
SUPPORTED_LANGUAGES = os.getenv("SUPPORTED_LANGUAGES", "fr,en").split(",")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "fr")

# Connexion Base de données
if DB_TYPE == "sqlite":
    DB_CONNECTION_STR = f"sqlite:///{PROJECT_ROOT / DB_NAME}"
else:
    DB_CONNECTION_STR = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Détection de langue
def detect_language(text: str) -> str:
    """Détecte la langue du texte."""
    if not text or not isinstance(text, str):
        return DEFAULT_LANGUAGE
    
    text_lower = text.lower()
    
    # Mots clés français
    french_keywords = {'le', 'la', 'les', 'un', 'une', 'des', 'je', 'tu', 'il', 'nous', 'vous', 'ils', 'est', 'sont'}
    # Mots clés anglais
    english_keywords = {'the', 'a', 'an', 'i', 'you', 'he', 'she', 'we', 'they', 'is', 'are', 'am'}
    
    french_count = sum(1 for word in french_keywords if word in text_lower)
    english_count = sum(1 for word in english_keywords if word in text_lower)
    
    if french_count > english_count:
        return "fr"
    elif english_count > french_count:
        return "en"
    else:
        return DEFAULT_LANGUAGE
