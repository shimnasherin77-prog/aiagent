from fastapi import FastAPI
from app.schemas import AskRequest, AskResponse
from app.agent import agent

app = FastAPI(
    title="AI Agent with RAG",
    version="1.0"
)

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    answer, sources = agent.decide_and_answer(
        query=request.query,
        session_id=request.session_id
    )

    return AskResponse(
        answer=answer,
        source=sources
    )