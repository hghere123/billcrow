📚 RAG-based FAQ Assistant










A lightweight Retrieval-Augmented Generation (RAG) system designed to answer questions from internal documents with high accuracy, low latency, and minimal hallucination.

🚀 Features
⚡ Fast semantic search using FAISS
🧠 Lightweight embedding model (CPU-friendly)
💸 Cost-efficient (LLM used only when necessary)
❌ Hallucination reduction via strict grounding
🔍 Transparent retrieval (context visible in UI)
🧠 Models Used
🔹 Embedding Model
sentence-transformers/all-MiniLM-L6-v2

✔ Lightweight
✔ Fast inference
✔ Strong semantic similarity

🔹 LLMs (via OpenRouter)
mistralai/mistral-7b-instruct:free
meta-llama/llama-2-7b-chat:free
google/flan-t5-xxl:free
Fallback: openrouter/auto

✔ No local hosting required
✔ Free inference options
✔ Unified API

🏗️ Architecture Diagram
5
🧩 System Workflow
User Query
     ↓
Embedding Model (MiniLM)
     ↓
FAISS Vector Search
     ↓
Top-K Relevant Chunks
     ↓
 ┌───────────────┐
 │ FAQ Shortcut? │──Yes──→ Direct Answer
 └───────┬───────┘
         ↓ No
   LLM (Strict Prompt)
         ↓
     Final Answer
📂 Document Pipeline
1️⃣ Chunking
Split by headings & paragraphs
Optional overlap
{
  "text": "chunk content",
  "source": "document filename",
  "metadata": {}
}
2️⃣ Embedding
Convert chunks into dense vectors using MiniLM
3️⃣ Indexing
Stored in:
vector_store.index (FAISS)
metadata.json
🔍 Retrieval Flow
Encode user query
Perform FAISS nearest-neighbor search
Retrieve Top-K chunks
Display:
Source
Similarity score
Context
🛡️ Anti-Hallucination Strategy
✅ 1. FAQ Shortcut
Detects Q: / A: patterns
Returns exact answer (no LLM)
✅ 2. Constrained LLM Prompt
Answer only from context
Max 2 sentences

If unsure →

I don’t know.
✅ 3. Transparency
Shows retrieved chunks
Displays final answer
Easy to verify grounding
🛠️ Tech Stack
Python
Sentence Transformers
FAISS
OpenRouter API
JSON
⚙️ Installation
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt
▶️ Usage
python app.py

Then:

Enter your query
View retrieved context
Get grounded answer
