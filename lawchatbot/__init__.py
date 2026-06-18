"""
LawChatbot package initialization.

This package provides reusable components for legal document retrieval and RAG-based QA.
"""

from .config import AppConfig, load_config
from .weaviate_client import initialize_weaviate_client
from .embedding import JinaEmbeddingWrapper
from .vectorstore import initialize_vector_store
from .retrievers import (
    initialize_semantic_retriever,
    initialize_bm25_retriever,
    initialize_hybrid_retriever,
    wrap_retriever_with_source
)
from .rag_chain import initialize_llm, build_rag_chain, run_rag_query
