from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # TODO: store references to store and llm_fn
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # TODO: retrieve chunks, build prompt, call llm_fn
        """
        Retrieves context from the store and generates an answer using the LLM.
        """
        # 1. Retrieve top-k relevant chunks
        results = self.store.search(question, top_k=top_k)
        
        if not results:
            context_text = "No relevant information found in the knowledge base."
        else:
            # Extract the text content from the retrieved records
            context_parts = [res["content"] for res in results]
            context_text = "\n\n".join(context_parts)

        # 2. Build a structured prompt
        # We provide the context to the LLM to ground its response in facts.
        prompt = (
            "You are a helpful assistant. Use the provided context to answer the question.\n"
            "If the answer isn't in the context, say you don't know based on the knowledge base.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        # 3. Call the LLM to generate an answer
        return self.llm_fn(prompt)
