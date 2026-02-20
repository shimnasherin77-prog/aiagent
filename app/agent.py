import os
from dotenv import load_dotenv
from openai import OpenAI
from app.memory import memory_store
from app.tools import rag_tool

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class AIAgent:

    def should_use_rag(self, context: str) -> bool:
        # If no meaningful context, skip RAG
        if context is None:
            return False
        if context.strip() == "":
            return False
        if len(context) < 20:
            return False
        return True

    def decide_and_answer(self, query: str, session_id: str):
        # Get chat history
        history = memory_store.get_history(session_id)

        # Step 1: Retrieve from local documents (NO web)
        rag_result = rag_tool(query)
        context = rag_result["context"]
        sources = rag_result["sources"]

        use_rag = self.should_use_rag(context)

        if use_rag:
            system_prompt = """
You are an AI assistant that answers using provided document context.
If context is relevant, base your answer on it.
Be clear and professional.
"""
            user_content = f"Context:\n{context}\n\nQuestion:\n{query}"
        else:
            system_prompt = """
You are a helpful AI assistant.
Answer general knowledge questions directly.
"""
            user_content = query
            sources = []  # No document used

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3
        )

        answer = response.choices[0].message.content

        # Save memory
        memory_store.add_message(session_id, "user", query)
        memory_store.add_message(session_id, "assistant", answer)

        return answer, sources

agent = AIAgent()