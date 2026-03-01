import streamlit as st
import sys
from pathlib import Path

# Ajout du chemin racine au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline
from src.config import config

# Initialisation du pipeline (en cache)
@st.cache_resource
def init_pipeline():
    pipeline = RAGPipeline()
    return pipeline

def main():
    st.set_page_config(
        page_title="Assistant PDF RAG",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Assistant PDF - RAG")
    st.markdown("Posez des questions sur vos documents PDF")
    
    # Initialisation
    pipeline = init_pipeline()
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("📊 Statistiques")
        stats = pipeline.get_stats()
        st.write(f"**Chunks:** {stats['total_chunks']}")
        st.write(f"**Modèle:** {stats['embedding_model']}")
        
        st.subheader("🔄 Indexation")
        if st.button("Réindexer les PDFs"):
            with st.spinner("Indexation en cours..."):
                pipeline.index_pdfs(force_reindex=True)
            st.success("✅ Indexation terminée!")
            st.rerun()
        
        st.divider()
        st.markdown("**Dossier PDF:**")
        st.code(str(config.PDF_DIR))
    
    # Zone de chat principale
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Affichage de l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Affichage des sources si disponibles
            if "sources" in message and message["sources"]:
                with st.expander("📖 Sources consultées"):
                    for i, source in enumerate(message["sources"][:3]):
                        st.caption(f"**Source {i+1}** (Page {source.metadata.get('page', '?')})")
                        st.text(source.content[:300] + "...")
    
    # Input utilisateur
    if prompt := st.chat_input("Posez votre question sur les PDFs"):
        # Ajout du message utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Génération de la réponse
        with st.chat_message("assistant"):
            with st.spinner("Recherche dans les PDFs..."):
                response = pipeline.query(prompt)
                st.markdown(response.answer)
                
                # Affichage des sources
                if response.sources:
                    with st.expander("📖 Sources utilisées"):
                        for i, source in enumerate(response.sources[:3]):
                            page = source.metadata.get('page', '?')
                            source_name = source.metadata.get('source', 'Inconnu')
                            st.caption(f"**Source {i+1}** - {source_name}, Page {page}")
                            st.text(source.content[:300] + "...")
        
        # Ajout de la réponse à l'historique
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources
        })

if __name__ == "__main__":
    main()