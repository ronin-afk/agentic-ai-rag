# Agentic AI RAG Chatbot

A document-grounded Retrieval-Augmented Generation (RAG) chatbot built using Python, LangGraph, LangChain, local embeddings, a local vector store, and FastAPI.

The chatbot is designed to answer questions strictly using information retrieved from the provided **Agentic AI eBook**.

---

## Objective

The objective of this project is to build a RAG-based chatbot that:

* Loads and processes the provided Agentic AI eBook.
* Splits the document into meaningful text chunks.
* Generates vector embeddings for the chunks.
* Stores the embeddings in a vector database.
* Retrieves the most relevant chunks for a user query.
* Uses LangGraph to orchestrate the retrieval and generation workflow.
* Generates answers grounded only in the retrieved document context.
* Refuses to answer when the required information is not available in the document.
* Returns the final answer, retrieved context, and confidence score through a FastAPI endpoint.

---

## Architecture

```text
                    Agentic AI PDF
                          |
                          v
                     PDF Loader
                          |
                          v
                    Text Chunking
                          |
                          v
                  Local Embeddings
                          |
                          v
                    Vector Store
                          |
                          |
User Query ---------------+
                          |
                          v
                   Query Embedding
                          |
                          v
                    Top-K Retrieval
                          |
                          v
                      LangGraph
                          |
                          v
                Grounded Generation
                          |
                          v
                       FastAPI
                          |
                          v
                   JSON Response
```

---

## RAG Workflow

The application follows the following pipeline:

```text
PDF
 |
 |-- Load document
 |
 |-- Extract page-level text
 |
 |-- Split text into chunks
 |
 |-- Generate embeddings
 |
 |-- Store vectors + metadata
 |
 v
Vector Store
 |
 |-- User Query
 |
 |-- Generate query embedding
 |
 |-- Similarity Search
 |
 |-- Retrieve Top-K chunks
 |
 v
LangGraph
 |
 |-- Retrieve Node
 |
 |-- Generate Node
 |
 v
Grounded Answer
 |
 v
FastAPI JSON Response
```

---

## Technologies

* Python
* LangGraph
* LangChain
* Sentence Transformers
* FAISS
* PyPDF
* FastAPI
* Uvicorn
* Pydantic

### Why local components?

The implementation uses local embedding and vector-store components so that the complete RAG pipeline can be executed without requiring paid API credits.

The architecture remains the same:

```text
Document
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector Store
   ↓
Retrieval
   ↓
LangGraph
   ↓
Grounded Generation
   ↓
FastAPI
```

---

## Project Structure

```text
agentic-ai-rag/
│
├── data/
│   └── Ebook-Agentic-AI.pdf
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion.py
│   └── graph.py
│
├── vectorstore/
│
├── app.py
├── test_queries.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Setup

## 1. Clone the Repository

```bash
git clone <repository-url>
cd agentic-ai-rag
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Add the Agentic AI PDF

Place the provided Agentic AI eBook inside the `data` directory:

```text
data/Ebook-Agentic-AI.pdf
```

The ingestion pipeline reads this document and creates the searchable vector representation.

---

# Data Ingestion

Run:

```bash
python -m src.ingestion
```

The ingestion pipeline performs the following operations:

1. Loads the PDF using PyPDF.
2. Extracts text from each page.
3. Splits the document into smaller chunks.
4. Preserves page information as metadata.
5. Generates vector embeddings for each chunk.
6. Stores the embeddings and metadata in the local vector store.

Example metadata associated with a chunk:

```text
{
    "text": "...document content...",
    "page": 5,
    "source": "Ebook-Agentic-AI.pdf"
}
```

---

# LangGraph Workflow

LangGraph is used to organize the RAG workflow into explicit processing nodes.

The current workflow is:

```text
START
  |
  v
Retrieve
  |
  v
Generate
  |
  v
END
```

### Retrieve Node

The retrieve node:

1. Receives the user's question.
2. Generates an embedding for the question.
3. Performs similarity search against the vector store.
4. Retrieves the most relevant document chunks.
5. Passes the retrieved context to the next node.

### Generate Node

The generate node:

1. Receives the user's question.
2. Receives the retrieved document context.
3. Generates an answer using the available context.
4. Avoids introducing information that is not supported by the retrieved document.

If the retrieved context does not contain enough information to answer the question, the chatbot returns:

```text
I cannot answer based on the provided document.
```

---

# FastAPI

Start the API using:

```bash
uvicorn app:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

# API

## POST `/chat`

The endpoint accepts a user question.

### Request

```json
{
  "query": "What is Agentic AI?"
}
```

### Response

```json
{
  "query": "What is Agentic AI?",
  "final_answer": "Agentic AI refers to AI systems that can reason, plan, and take actions toward achieving a goal.",
  "retrieved_context_chunks": [
    {
      "text": "Relevant text retrieved from the Agentic AI eBook...",
      "page": 5
    },
    {
      "text": "Another relevant section from the document...",
      "page": 6
    }
  ],
  "confidence_score": 0.87
}
```

---

# Response Fields

### `query`

The original question submitted by the user.

### `final_answer`

The final response generated by the chatbot using the retrieved document context.

### `retrieved_context_chunks`

The document chunks retrieved from the vector store for the given query.

Each chunk contains relevant source text and page information.

### `confidence_score`

A retrieval relevance score representing how strongly the retrieved context matches the user's query.

The score is based on the similarity of the retrieved vectors and should be interpreted as a retrieval-confidence indicator rather than a guaranteed measure of factual correctness.

---

# Example Queries

The following questions can be used to validate the chatbot.

### 1. Agentic AI Definition

```text
What is the core definition of Agentic AI as outlined in the eBook?
```

### 2. Architecture

```text
What are the main architectural components required to build agentic systems?
```

### 3. Industry Use Cases

```text
What real-world industry use cases for Agentic AI are discussed?
```

### 4. Traditional Chatbots

```text
How does Agentic AI differ from traditional generative AI chatbots?
```

### 5. Challenges

```text
What key challenges or limitations are mentioned?
```

### 6. Out-of-Scope Question

```text
What is the capital of France?
```

The chatbot should not use general world knowledge to answer an out-of-scope question.

Instead, it should respond that the information is not available in the provided document.

---

# Testing

Run:

```bash
python test_queries.py
```

The test script sends the sample questions to the FastAPI `/chat` endpoint and displays:

* User query
* Final answer
* Retrieved context
* Confidence score

The test cases include both document-related questions and an out-of-scope question to verify document grounding.

---

# Grounding Strategy

The chatbot is designed to minimize hallucination by restricting answer generation to retrieved document context.

The generation prompt instructs the system to:

```text
Use only the provided context to answer the question.

Do not use external or general knowledge.

Do not invent information.

If the answer cannot be found in the provided context,
state that the information is not available in the document.
```

This ensures that the chatbot behaves as a document-grounded RAG system rather than a general-purpose chatbot.

---

# Key Design Decisions

## Chunking

The PDF is divided into smaller overlapping chunks so that relevant sections can be retrieved independently.

Chunking improves retrieval because an entire document does not need to be passed to the generation stage.

## Metadata

Each chunk stores metadata such as:

* Source document
* Page number
* Original text

This makes the retrieved context traceable to the source document.

## Top-K Retrieval

The system retrieves the most relevant chunks for every query.

This reduces the amount of irrelevant information passed to the generation stage.

## LangGraph

LangGraph provides an explicit workflow for:

```text
Question
   ↓
Retrieve
   ↓
Generate
   ↓
Answer
```

This makes the RAG pipeline modular and easier to extend with additional nodes such as relevance grading or query refinement.

---

# Limitations

The current implementation is intentionally focused on the provided Agentic AI eBook.

It does not:

* Search the internet
* Use external knowledge sources
* Answer questions outside the provided document
* Support multiple knowledge bases
* Provide authentication
* Maintain long-term conversation memory

These constraints help maintain document grounding.

---

# Future Improvements

Possible improvements include:

* Relevance grading before generation
* Query rewriting
* Conversation memory
* Source citation in generated answers
* Multiple document support
* Hybrid keyword + vector search
* Reranking retrieved chunks
* Additional evaluation metrics
* Production deployment

---

# Conclusion

This project demonstrates a complete Retrieval-Augmented Generation pipeline using:

```text
PDF
 ↓
Chunking
 ↓
Embeddings
 ↓
Vector Search
 ↓
LangGraph Retrieval
 ↓
Grounded Generation
 ↓
FastAPI
 ↓
Structured JSON Response
```

