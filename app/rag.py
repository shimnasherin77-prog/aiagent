import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

DATA_PATH = "data"

class RAGSystem:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = self._load_and_embed()

    def _load_documents(self):
        texts = []
        metadata = []

        if not os.path.exists(DATA_PATH):
            return texts, metadata

        for file in os.listdir(DATA_PATH):
            if file.endswith(".txt"):
                file_path = os.path.join(DATA_PATH, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    texts.append(content)
                    metadata.append({"source": file})

        return texts, metadata

    def _load_and_embed(self):
        texts, metadata = self._load_documents()

        if not texts:
            return None

        splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = []
        metadatas = []

        for text, meta in zip(texts, metadata):
            split_texts = splitter.split_text(text)
            chunks.extend(split_texts)
            metadatas.extend([meta] * len(split_texts))

        vectorstore = FAISS.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            metadatas=metadatas
        )

        return vectorstore

    def retrieve(self, query: str, k: int = 3):
        if self.vectorstore is None:
            return "", []

        docs = self.vectorstore.similarity_search(query, k=k)

        context = "\n".join([doc.page_content for doc in docs])
        sources = list(set([doc.metadata["source"] for doc in docs]))

        return context, sources

rag_system = RAGSystem()