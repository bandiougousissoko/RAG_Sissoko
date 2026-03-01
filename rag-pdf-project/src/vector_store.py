from typing import List, Optional, Dict, Any
import pickle
import json
from pathlib import Path
import logging
import numpy as np
from src.models import Chunk
from src.embeddings import EmbeddingManager
from src.config import config

logger = logging.getLogger(__name__)

class VectorStore:
    """Interface unifiée pour différentes bases vectorielles"""
    
    def __init__(self, persist_dir: Path = None, embedding_manager: EmbeddingManager = None):
        self.persist_dir = persist_dir or config.VECTOR_STORE_DIR
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.chunks: List[Chunk] = []
        self.embeddings_matrix: Optional[np.ndarray] = None
        
        # Création du dossier si nécessaire
        self.persist_dir.mkdir(parents=True, exist_ok=True)
    
    def add_chunks(self, chunks: List[Chunk], embeddings: Optional[List[List[float]]] = None):
        """Ajoute des chunks à la base avec leurs embeddings"""
        if embeddings is None:
            # Génération automatique des embeddings
            texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_manager.embed_texts(texts)
        
        # Stockage des chunks
        start_idx = len(self.chunks)
        self.chunks.extend(chunks)
        
        # Construction de la matrice d'embeddings
        new_embeddings = np.array(embeddings)
        if self.embeddings_matrix is None:
            self.embeddings_matrix = new_embeddings
        else:
            self.embeddings_matrix = np.vstack([self.embeddings_matrix, new_embeddings])
        
        logger.info(f"✅ {len(chunks)} chunks ajoutés (total: {len(self.chunks)})")
    
    def similarity_search(self, query: str, k: int = None) -> List[tuple]:
        """Recherche les chunks les plus similaires à la requête"""
        if k is None:
            k = config.TOP_K_RESULTS
        
        if not self.chunks or self.embeddings_matrix is None:
            logger.warning("⚠️ Base vectorielle vide")
            return []
        
        # Embedding de la requête
        query_embedding = self.embedding_manager.embed_text(query)
        query_vector = np.array(query_embedding)
        
        # Calcul des similarités (cosine similarity)
        similarities = np.dot(self.embeddings_matrix, query_vector) / (
            np.linalg.norm(self.embeddings_matrix, axis=1) * np.linalg.norm(query_vector)
        )
        
        # Récupération des k meilleurs résultats
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            similarity = float(similarities[idx])
            results.append((chunk, similarity))
        
        return results
    
    def save(self):
        """Sauvegarde la base vectorielle sur disque"""
        # Sauvegarde des chunks en JSON
        chunks_file = self.persist_dir / "chunks.json"
        chunks_data = [
            {
                "content": c.content,
                "metadata": c.metadata,
                "chunk_id": c.chunk_id
            }
            for c in self.chunks
        ]
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        # Sauvegarde des embeddings en numpy
        embeddings_file = self.persist_dir / "embeddings.npy"
        if self.embeddings_matrix is not None:
            np.save(embeddings_file, self.embeddings_matrix)
        
        logger.info(f"💾 Base sauvegardée dans {self.persist_dir}")
    
    def load(self):
        """Charge une base vectorielle depuis le disque"""
        chunks_file = self.persist_dir / "chunks.json"
        embeddings_file = self.persist_dir / "embeddings.npy"
        
        if not chunks_file.exists() or not embeddings_file.exists():
            logger.warning("⚠️ Aucune base existante trouvée")
            return
        
        # Chargement des chunks
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        self.chunks = []
        for data in chunks_data:
            chunk = Chunk(
                content=data["content"],
                metadata=data["metadata"],
                chunk_id=data["chunk_id"]
            )
            self.chunks.append(chunk)
        
        # Chargement des embeddings
        self.embeddings_matrix = np.load(embeddings_file)
        
        logger.info(f"📂 Base chargée: {len(self.chunks)} chunks")