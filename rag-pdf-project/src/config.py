import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    """Configuration centralisée de l'application"""
    
    # Chemins
    BASE_DIR = Path(__file__).parent.parent
    PDF_DIR = BASE_DIR / "data" / "pdfs"
    VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"
    
    # Paramètres de découpage
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Modèles
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "gpt-3.5-turbo"
    
    # RAG
    TOP_K_RESULTS = 3
    TEMPERATURE = 0.0
    
    # API Keys (à mettre dans .env)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    
    # Base vectorielle
    VECTOR_DB_TYPE = "chroma"  # ou "faiss"
    
    def __post_init__(self):
        """Création automatique des dossiers nécessaires"""
        self.PDF_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Instance globale de configuration
config = Config()