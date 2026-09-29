from typing import TypedDict, List

from langgraph.graph import StateGraph, START, END

from langchain_openai import (
    OpenAIEmbeddings,
    ChatOpenAI,
)

from pinecone import Pinecone

from src.config import (
    OPENAI_EMBEDDING_MODEL,
    OPENAI_CHAT_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)


class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    confidence_score: float


embeddings = OpenAIEmbeddings(
    model=OPENAI_EMBEDDING_MODEL
)

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(PINECONE_INDEX_NAME)

llm = ChatOpenAI(
    model=OPENAI_CHAT_MODEL,
    temperature=0,
)


def retrieve_node(state: AgentState):
    """
    Retrieve the most relevant chunks from Pinecone.
    """

    question = state["question"]

    query_vector = embeddings.embed_query(question)

    result = index.query(
        vector=query_vector,
        top_k=4,
        include_metadata=True,
        include_values=False,
    )

    matches = result.get("matches", [])

    context = []

    for match in matches:
        metadata = match.get("metadata", {})

        text = metadata.get("text", "")
        page = metadata.get("page", "unknown")

        score = match.get("score", 0.0)

        if text:
            context.append(
                f"[Page {page} | similarity={score:.4f}]\n{text}"
            )

    # Highest similarity score
    confidence = 0.0

    if matches:
        confidence = float(matches[0].get("score", 0.0))

    return {
        "context": context,
        "confidence_score": confidence,
    }


def generate_node(state: AgentState):
    """
    Generate an answer using ONLY the retrieved context.
    """

    question = state["question"]
    context = state["context"]
    retrieval_score = state["confidence_score"]

    if not context:
        return {
            "answer": (
                "I cannot answer based on the provided document."
            )
        }

    context_text = "\n\n".join(context)

    prompt = f"""
You are a strict document-grounded AI assistant.

Your only source of truth is the context provided below.

Rules:
1. Answer ONLY using information present in the context.
2. Do not use your general knowledge.
3. Do not invent facts.
4. If the context does not contain enough information to answer
   the question, say exactly:
   "I cannot answer based on the provided document."
5. Keep the answer concise but informative.
6. Do not mention information that is not supported by the context.

Retrieved Context:
------------------
{context_text}
------------------

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    answer = response.content.strip()

    return {
        "answer": answer
    }


def build_graph():
    """
    Build the LangGraph RAG workflow.
    """

    workflow = StateGraph(AgentState)

    workflow.add_node(
        "retrieve",
        retrieve_node
    )

    workflow.add_node(
        "generate",
        generate_node
    )

    workflow.add_edge(
        START,
        "retrieve"
    )

    workflow.add_edge(
        "retrieve",
        "generate"
    )

    workflow.add_edge(
        "generate",
        END
    )

    return workflow.compile()