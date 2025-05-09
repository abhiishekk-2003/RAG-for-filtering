# formatter.py
def format_context(results):
    chunks = [hit["payload"]["text"] for hit in results]
    return "\n---\n".join(chunks)
