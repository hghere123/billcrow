Indecimal RAG Chatbot
This project is a Retrieval-Augmented Generation (RAG) chatbot that answers questions about Indecimal’s construction services using only the provided internal documents. It combines document chunking, semantic search with FAISS, and an LLM accessed via OpenRouter, wrapped in a simple custom UI (FastAPI or Streamlit, depending on your setup).

1. Project Overview
The chatbot:

Ingests the supplied Markdown documents about Indecimal.

Chunks them into semantically meaningful segments.

Embeds each chunk using a SentenceTransformers embedding model.

Indexes embeddings in a local FAISS vector store for fast similarity search.

For each user query:

Retrieves the top‑k most relevant chunks.

Uses an LLM (via OpenRouter) to generate an answer grounded only in those chunks.

Displays:

The retrieved context chunks.

The final generated answer.

This satisfies the requirements for document processing, vector indexing, grounded answer generation, and transparency.

2. Architecture
Components
Embedding model:

sentence-transformers/all-MiniLM-L6-v2 (local, open‑source).

Chosen because it is lightweight, fast on CPU, and has strong performance for semantic similarity on short to medium text.

Vector store & retrieval:

FAISS (L2 / inner‑product index) for local semantic search over chunk embeddings.

Retrieval: top‑k (default 3) chunks per query using FAISS’s nearest‑neighbor search.

LLM for answer generation:

Primary: free OpenRouter models such as

mistralai/mistral-7b-instruct:free

meta-llama/llama-2-7b-chat:free

Fallback: openrouter/auto

Chosen because they provide a reasonable balance of quality and latency through a single API, without needing to host a model.

Backend logic:

RAGPipeline class:

retrieve(query, top_k) → returns ranked chunks with text, source, and score.

generate_answer(query, contexts) → returns an answer tightly grounded in the retrieved contexts, with FAQ‑style extraction when possible and conservative behavior (“I don’t know”) when the context is insufficient.

Frontend:

Either:

A FastAPI JSON endpoint consumed by a custom web UI, or

A Streamlit app (app.py) providing a chat‑style interface.

The UI shows both the final answer and the source chunks used.

3. Document Chunking & Embedding
Document processing
Read all provided .md documents (e.g., doc1.md, doc2.md, doc3.md).

Normalize text (strip headers, remove extra whitespace if needed).

Chunking strategy:

Split by sections and paragraphs (e.g., headers or double newlines).

Optionally apply a text splitter (e.g., langchain-text-splitters) with:

Chunk size ~ 500–800 characters

Overlap ~ 50–100 characters

Each chunk stores:

text: the raw chunk content

source: document name and an optional section reference

Generate embeddings:

Use SentenceTransformer('all-MiniLM-L6-v2').

Compute a vector for each chunk and store them in memory for indexing.

Vector indexing
Build a FAISS index:

Convert embeddings to float32 NumPy array.

Use faiss.IndexFlatIP or IndexFlatL2 depending on similarity metric.

Add all chunk embeddings to the index.

Save:

vector_store.index (FAISS index)

metadata.json (list of {text, source} for each vector)

The ingestion is typically handled by an ingest.py script, which must be run once before querying.

4. Retrieval and Grounded Answer Generation
Retrieval (RAGPipeline.retrieve)
Given a user query:

Encode query to an embedding with the same SentenceTransformer model.

Search FAISS for top‑k nearest neighbors.

For each hit, return:

text

source

similarity score

These chunks are passed into the answer generation step.

Grounded answer generation (RAGPipeline.generate_answer)
The pipeline enforces grounding in two layers:

FAQ pattern shortcut (zero hallucination):

If any retrieved context contains explicit Q: / A: blocks and the question text closely matches a Q:, return the corresponding A: line directly without calling the LLM.

This is used, for example, for:

“How does Indecimal reduce hidden surprises in pricing?” → returns the exact FAQ answer from doc1.

LLM‑based reasoning with strict prompt:

If no FAQ answer is found, the system constructs a CONTEXT string by joining the retrieved chunks with their sources.

The prompt instructs the LLM to:

First think about 1–3 relevant sentences in the context.

Answer in at most 2 short sentences.

Use only the context; if insufficient, say exactly "I don't know.".

Never add external knowledge or speculation.

The system uses a conservative temperature (e.g., 0.1) to reduce variability and hallucinations.

If all model calls fail (e.g., network or API issues), a descriptive error is returned.

5. Transparency & Explainability
The UI clearly exposes both:

Retrieved context:

For the last user question, the app re‑runs retrieval and displays:

Source name (e.g., doc1.md – FAQs)

The chunk text used as context.

Final answer:

Shown inline in the chat area as the assistant’s response.

This satisfies the requirement to show both the supporting chunks and the generated answer.

6. Running the Project Locally
6.1. Prerequisites
Python 3.9+

Git

An OpenRouter API key: https://openrouter.ai/keys

6.2. Setup
bash
# Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
Create a .env file in the same folder as rag_core.py:

text
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxx
6.3. Build the index (one-time)
bash
python ingest.py
This script should:

Load documents from a docs/ or similar folder.

Chunk them and compute embeddings.

Save data/index/vector_store.index and data/index/metadata.json.

6.4. Start the UI
Option A: Streamlit app
bash
streamlit run app.py
Then open the URL shown in the terminal (usually http://localhost:8501).

Option B: FastAPI backend + custom frontend
bash
uvicorn main:app --reload
Backend: exposes an endpoint like /rag/query that wraps RAGPipeline.

Frontend: a simple web client (React, vanilla HTML/JS, etc.) that posts the user query and displays the answer plus context.

7. Deployment
You can deploy in several ways; two common options:

7.1. Streamlit Community Cloud
Push the repo to GitHub.

Ensure app.py and requirements.txt are at the repo root.

Add OPENROUTER_API_KEY in Streamlit Secrets.

Create a new app on https://share.streamlit.io and point it to app.py.

7.2. Render.com (FastAPI)
Use main.py with a FastAPI app and RAGPipeline.

Configure Render Web Service:

Build command: pip install -r requirements.txt

Start command: uvicorn main:app --host 0.0.0.0 --port $PORT

Set environment variable OPENROUTER_API_KEY in Render dashboard.

8. Optional Enhancements (Not Implemented / Ideas)
These are suggested extensions aligned with the bonus requirements:

Local open‑source LLM (e.g., via Ollama or Hugging Face):

Replace/augment OpenRouter calls with a local 2B–3B model.

Compare answer quality, latency, and groundedness.

Quality analysis:

Create 8–15 test questions from the documents (e.g., pricing, quality checks, warranties, payment flow).

For each:

Inspect retrieved chunks for relevance.

Check answers for hallucinations and completeness.

Summarize findings (e.g., “90% of answers fully grounded, 2/10 questions returned ‘I don’t know,’ etc.”).

Brief summaries and metrics can be added here once you run such an evaluation.

9. Files Overview
rag_core.py – RAGPipeline: retrieval + grounded answer generation.

ingest.py – document loading, chunking, embedding, and FAISS index creation.

app.py – Streamlit UI or API wrapper that exposes the chatbot.

data/index/vector_store.index – FAISS index file.

data/index/metadata.json – list of chunk metadata (text, source).

requirements.txt – Python dependencies.

README.md – this document.

u**. You will need to commit the `FAISS` index folder, or configure the host to run `ingest.py` as a build command. Set the `OPENROUTER_API_KEY` as an environment variable in the dashboard.
