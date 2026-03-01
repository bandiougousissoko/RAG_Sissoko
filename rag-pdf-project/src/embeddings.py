from typing import List, Optional
import logging
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from src.config import config

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Gestionnaire d'embeddings avec support multi-modèles"""
    
    def __init__(self, model_name: Optional[str] = None, model_type: str = "huggingface"):
        self.model_name = model_name or config.EMBEDDING_MODEL
        
        if model_type == "huggingface":
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        elif model_type == "openai":
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002"
            )
        else:
            raise ValueError(f"Type de modèle inconnu: {model_type}")
        
        logger.info(f"✅ Modèle d'embedding chargé: {self.model_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """Génère l'embedding pour un texte simple"""
        return self.embeddings.embed_query(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Génère les embeddings pour une liste de textes"""
        return self.embeddings.embed_documents(texts)
    
    def embed_chunks(self, chunks) -> List[List[float]]:
        """Génère les embeddings pour une liste de chunks"""
        texts = [chunk.content for chunk in chunks]
        return self.embed_texts(texts)