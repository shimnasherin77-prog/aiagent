from app.rag import rag_system

def rag_tool(query: str):
    context, sources = rag_system.retrieve(query)
    return {
        "context": context,
        "sources": sources
    }