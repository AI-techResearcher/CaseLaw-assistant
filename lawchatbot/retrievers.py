from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.bm25 import BM25Retriever
from langchain_core.documents import Document
from typing import Any
from langchain.retrievers import EnsembleRetriever

def initialize_semantic_retriever(vectorstore, config) -> BaseRetriever:
    """
    Initialize semantic retriever from the Weaviate vector store.
    """
    print("🤖 Initializing semantic retriever...")
    semantic_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.semantic_k}
    )
    print(f"✅ Semantic retriever initialized with k={config.semantic_k}")
    return semantic_retriever

def initialize_bm25_retriever(client, config) -> BM25Retriever:
    """
    Initialize BM25 keyword retriever from Weaviate documents.
    """
    print("🔍 Fetching documents for BM25 initialization...")
    collection = client.collections.get(config.weaviate_class)
    response = collection.query.fetch_objects(
        limit=1000,
        return_properties=config.metadata_attributes + [config.text_key]
    )
    documents = []
    for obj in response.objects:
        text = obj.properties.get(config.text_key, "")
        metadata = {k: v for k, v in obj.properties.items() if k != config.text_key}
        metadata["source"] = getattr(obj, "uuid", "unknown")
        documents.append(Document(page_content=text, metadata=metadata))
    if not documents:
        raise ValueError("No documents retrieved from Weaviate — cannot initialize BM25.")
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    print(f"📄 Retrieved {len(documents)} documents for BM25.")
    bm25_retriever = BM25Retriever.from_texts(
        texts=texts,
        metadatas=metadatas,
        k=config.bm25_k
    )
    print(f"✅ BM25 retriever initialized with k={config.bm25_k}")
    return bm25_retriever

def initialize_hybrid_retriever(semantic_retriever, bm25_retriever, alpha: float = 0.5) -> BaseRetriever:
    """
    Combine semantic and BM25 retrievers into a hybrid retriever using EnsembleRetriever.
    """
    print("🔗 Combining semantic and BM25 retrievers into hybrid retriever...")
    hybrid_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[alpha, 1 - alpha]
    )
    print(f"✅ Hybrid retriever initialized with alpha={alpha}")
    return hybrid_retriever

from pydantic import PrivateAttr

def wrap_retriever_with_source(retriever):
    """
    Wraps a retriever so every returned document has a 'source' key in metadata.
    """
    class RetrieverWithSource(BaseRetriever):
        base_retriever: Any

        def __init__(self, base_retriever):
            super().__init__()
            self.base_retriever = base_retriever

        def _get_relevant_documents(self, query):
            docs = self.base_retriever.get_relevant_documents(query)
            for doc in docs:
                if "source" not in doc.metadata:
                    if hasattr(doc, "id"):
                        doc.metadata["source"] = doc.id
                    elif hasattr(doc, "uuid"):
                        doc.metadata["source"] = doc.uuid
                    else:
                        doc.metadata["source"] = "unknown"
            return docs

    return RetrieverWithSource(retriever)