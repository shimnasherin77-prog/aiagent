from collections import defaultdict

class SessionMemory:
    def __init__(self):
        self.store = defaultdict(list)

    def get_history(self, session_id: str):
        return self.store[session_id]

    def add_message(self, session_id: str, role: str, content: str):
        self.store[session_id].append({
            "role": role,
            "content": content
        })

memory_store = SessionMemory()