from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.models import Document, Chunk
from src.config import config

class TextChunker:
    """Découpe les documents en chunks optimisés pour RAG"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        """Transforme une liste de documents en chunks"""
        chunks = []
        
        for doc in documents:
            # Découpage du contenu
            split_texts = self.splitter.split_text(doc.content)
            
            # Création des chunks avec métadonnées
            for i, text in enumerate(split_texts):
                chunk_metadata = doc.metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(split_texts),
                    "chunk_size": len(text)
                })
                
                chunk = Chunk(
                    content=text,
                    metadata=chunk_metadata,
                    chunk_id=f"{doc.doc_id}_{i}"
                )
                chunks.append(chunk)
        
        return chunks
    
    def split_text(self, text: str) -> List[str]:
        """Découpe simplement un texte sans métadonnées"""
        return self.splitter.split_text(text)