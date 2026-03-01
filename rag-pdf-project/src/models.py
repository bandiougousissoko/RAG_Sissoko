from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
from uuid import uuid4

@dataclass
class Document:
    """Représente un document chargé"""
    content: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = field(default_factory=lambda: str(uuid4()))
    
@dataclass
class Chunk:
    """Représente un morceau de document"""
    content: str
    metadata: dict
    chunk_id: str
    embedding: Optional[List[float]] = None
    
@dataclass
class Query:
    """Représente une requête utilisateur"""
    text: str
    timestamp: datetime = field(default_factory=datetime.now)
    query_id: str = field(default_factory=lambda: str(uuid4()))
    
@dataclass
class RetrievalResult:
    """Résultat de la recherche"""
    chunks: List[Chunk]
    similarities: List[float]
    query: Query
    
@dataclass 
class RAGResponse:
    """Réponse finale du système RAG"""
    answer: str
    sources: List[Chunk]
    query: Query
    generation_time: float