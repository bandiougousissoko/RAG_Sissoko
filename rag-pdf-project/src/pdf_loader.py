from pathlib import Path
from typing import List, Optional
import logging
from langchain_community.document_loaders import PyPDFLoader
from src.models import Document
from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFLoader:
    """Gestionnaire de chargement de fichiers PDF"""
    
    def __init__(self, pdf_dir: Optional[Path] = None):
        self.pdf_dir = pdf_dir or config.PDF_DIR
        
    def load_single_pdf(self, pdf_path: Path) -> List[Document]:
        """Charge un seul fichier PDF"""
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            
            # Conversion vers notre modèle Document
            docs = []
            for page_num, doc in enumerate(documents):
                metadata = {
                    "source": pdf_path.name,
                    "page": page_num + 1,
                    "total_pages": len(documents),
                    "path": str(pdf_path)
                }
                docs.append(Document(
                    content=doc.page_content,
                    metadata=metadata
                ))
            
            logger.info(f"✅ PDF chargé: {pdf_path.name} - {len(docs)} pages")
            return docs
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement {pdf_path}: {e}")
            return []
    
    def load_all_pdfs(self) -> List[Document]:
        """Charge tous les PDFs du dossier configuré"""
        all_documents = []
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"⚠️ Aucun PDF trouvé dans {self.pdf_dir}")
            return []
        
        logger.info(f"📚 {len(pdf_files)} PDFs trouvés")
        
        for pdf_file in pdf_files:
            docs = self.load_single_pdf(pdf_file)
            all_documents.extend(docs)
            
        logger.info(f"✅ Total: {len(all_documents)} pages chargées")
        return all_documents