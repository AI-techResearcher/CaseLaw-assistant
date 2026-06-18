import os

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate

# Default OpenAI model; override via the OPENAI_MODEL environment variable.
DEFAULT_MODEL = "gpt-4o-mini"


def initialize_llm() -> ChatOpenAI:
    """
    Initialize the OpenAI chat LLM for RAG.

    Credentials and model are read from the environment so that no secret is
    ever committed to source control:

      - OPENAI_API_KEY (required) — your OpenAI API key
      - OPENAI_MODEL   (optional) — model id, defaults to gpt-4o-mini
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY: set it in the environment / .env file."
        )

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        api_key=api_key,
        temperature=0,
        max_tokens=8192,
    )

RAG_PROMPT = PromptTemplate.from_template(
    """You are an expert legal assistant. Use the provided context to answer the user question at the end.

If you use any document, cite it in the format: [source_name] with metadata (e.g., URL or case_id).

Be accurate, concise, and include citations for facts.

Context:
{context}

Question:
{question}

Answer (with citations):"""
)

def build_rag_chain(llm: ChatOpenAI, retriever) -> RetrievalQAWithSourcesChain:
    """
    Build a RetrievalQAWithSourcesChain using the provided LLM and retriever.
    """
    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": RAG_PROMPT,
            "document_variable_name": "context"
        },
        return_source_documents=True  # <-- Set this to True
    )

def run_rag_query(
    rag_chain: RetrievalQAWithSourcesChain,
    query: str,
    show_sources: bool = True  # Default to True
) -> str:
    """
    Run a RAG query and print the answer and relevant context.
    """
    response = rag_chain.invoke({"question": query})
    answer = response["answer"]
    documents = response.get("source_documents", [])
    # Optionally, you can return the answer and context as a dict if needed
    return answer