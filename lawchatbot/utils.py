def retrieve_documents(retriever, query: str):
    """
    Query a retriever and print the number of retrieved documents.
    """
    print(f"🔎 Querying retriever with: {query}")
    results = retriever.get_relevant_documents(query)
    print(f"📄 Retrieved {len(results)} documents.")
    return results