from weaviate import WeaviateClient, connect_to_weaviate_cloud, auth
from lawchatbot.config import AppConfig

def initialize_weaviate_client(config: AppConfig) -> WeaviateClient:
    """
    Initialize and return a Weaviate client connection.

    Args:
        config (AppConfig): Configuration object.

    Returns:
        WeaviateClient: A connected Weaviate client instance.
    """
    print("🔗 Connecting to Weaviate (cloud)...")
    client = connect_to_weaviate_cloud(
        cluster_url=config.weaviate_url,
        auth_credentials=auth.AuthApiKey(api_key=config.weaviate_api_key),
        # headers={"X-OpenAI-Api-Key": config.openai_api_key}  # Uncomment if needed
    )
    print("✅ Weaviate client initialized.")
    return client