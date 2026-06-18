from langchain_weaviate import WeaviateVectorStore as LCWeaviate
from lawchatbot.embedding import JinaEmbeddingWrapper
from lawchatbot.config import AppConfig

def initialize_vector_store(client, config: AppConfig) -> LCWeaviate:
    """
    Initialize LangChain Weaviate vector store.

    Args:
        client: A connected Weaviate client.
        config (AppConfig): Configuration object.

    Returns:
        LCWeaviate: LangChain-compatible Weaviate vector store.
    """
    print("📦 Initializing vector store...")

    embedder = JinaEmbeddingWrapper(device="cuda" if hasattr(config, "cuda") and config.cuda else "cpu")

    vectorstore = LCWeaviate(
        client=client,
        index_name=config.weaviate_class,
        text_key=config.text_key,
        attributes=config.metadata_attributes,
        embedding=embedder
    )

    print(f"✅ Vector store ready for class: {config.weaviate_class}")
    return vectorstore