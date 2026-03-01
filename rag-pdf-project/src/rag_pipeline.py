from typing import List, Optional
import logging
import time
from src.pdf_loader import PDFLoader
from src.text_splitter import TextChunker
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore
from src.generator import ResponseGenerator
from src.models import RAGResponse, Query
from src.config import config

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Orchestrateur du pipeline RAG complet"""
    
    def __init__(self):
        self.loader = PDFLoader()
        self.chunker = TextChunker()
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore(embedding_manager=self.embedding_manager)
        self.generator = ResponseGenerator()
        
        # Tentative de chargement d'une base existante
        self.vector_store.load()
    
    def index_pdfs(self, force_reindex: bool = False):
        """Indexe tous les PDFs dans la base vectorielle"""
        if not force_reindex and len(self.vector_store.chunks) > 0:
            logger.info("📚 Base déjà indexée, utilisation de l'existant")
            return
        
        # Chargement des PDFs
        documents = self.loader.load_all_pdfs()
        if not documents:
            logger.error("❌ Aucun document à indexer")
            return
        
        # Découpage en chunks
        chunks = self.chunker.split_documents(documents)
        
        # Génération des embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_manager.embed_texts(texts)
        
        # Stockage
        self.vector_store.add_chunks(chunks, embeddings)
        
        # Sauvegarde
        self.vector_store.save()
        
        logger.info(f"✅ Indexation terminée: {len(chunks)} chunks")
    
    def query(self, question: str) -> RAGResponse:
        """Exécute une requête complète"""
        start_time = time.time()
        
        # Création de la requête
        query = Query(text=question)
        
        # Recherche des chunks pertinents
        results = self.vector_store.similarity_search(question)
        chunks = [chunk for chunk, _ in results]
        
        if not chunks:
            return RAGResponse(
                answer="Je n'ai pas trouvé d'information pertinente dans les documents.",
                sources=[],
                query=query,
                generation_time=time.time() - start_time
            )
        
        # Génération de la réponse
        answer = self.generator.generate(question, chunks)
        
        # Création de la réponse
        response = RAGResponse(
            answer=answer,
            sources=chunks,
            query=query,
            generation_time=time.time() - start_time
        )
        
        logger.info(f"✅ Réponse générée en {response.generation_time:.2f}s")
        return response
    
    def get_stats(self) -> dict:
        """Retourne des statistiques sur le système"""
        return {
            "total_chunks": len(self.vector_store.chunks),
            "embedding_model": self.embedding_manager.model_name,
            "chunk_size": self.chunker.chunk_size,
            "top_k": config.TOP_K_RESULTS
        }