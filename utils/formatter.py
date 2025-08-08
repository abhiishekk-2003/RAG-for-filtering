# utils/formatter.py
def format_context(results):
    """Format search results into a context string for the LLM"""
    if not results:
        return "No relevant context found."
    
    chunks = []
    for hit in results:
        if "payload" in hit and "text" in hit["payload"]:
            chunks.append(hit["payload"]["text"])
    
    return "\n---\n".join(chunks)