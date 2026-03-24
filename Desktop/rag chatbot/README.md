# Mini RAG Pipeline

A comprehensive Retrieval-Augmented Generation (RAG) system built for answering questions grounded in internal documents. This project consists of a Python FastAPI backend, a custom React chatbot frontend, and an automated ingestion pipeline.
<img width="1713" height="810" alt="image" src="https://github.com/user-attachments/assets/f6417840-307e-4b0c-a619-2748b2145053" />
<img width="1766" height="820" alt="image" src="https://github.com/user-attachments/assets/3a0bbcb9-0e7e-4fc5-a474-bc59046e52c5" />

# References
Most precise answer comes when you kep the size of the chunk like this.


## 🏗 Architecture & Tech Stack

- **Backend Framework:** FastAPI (Python) - *Chosen for its speed, automatic Swagger UI documentation, and modern async support.*
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` - *Chosen because it runs locally, is free, fast, and generates high-quality semantic embeddings.*
- **Vector Store:** FAISS (Facebook AI Similarity Search) - *Chosen for rapid top-k nearest neighbor retrieval over numpy arrays.*
- **LLM for Generation:** OpenRouter (`meta-llama/llama-3-8b-instruct:free` or `google/gemma-7b-it:free`) - *Chosen for access to state-of-the-art models without requiring local GPU resources, while maintaining an OpenAI-compatible API.*
- **Frontend:** React + Vite - *Provides a sleek, modern, dynamic UI with expandable source citations.*

## 📖 How It Works

1. **Document Ingestion & Chunking (`backend/ingest.py`):**
   - Documents are downloaded programmatically from Google Drive using `gdown`.
   - Text is extracted using `pypdf` (and elegantly falls back to a plain text reader if the file is a text document disguised as a PDF).
   - Text is split using `RecursiveCharacterTextSplitter` (chunk size: 1000, overlap: 200). The 200-character overlap prevents sentences and meaning from being unnaturally severed between chunks.
2. **Retrieval (`backend/rag_core.py`):**
   - The user query is converted into a vector embedding using the same SentenceTransformer model.
   - FAISS performs an L2 distance search mathematically comparing the query vector against all chunk vectors, returning the top-3 most similar chunks.
3. **Grounded Generation:**
   - The retrieved chunks are formatted into a strict prompt template.
   - **Enforcing Grounding:** The prompt specifically instructs the model and system: *"If the answer is not contained in the context, explicitly say 'I don't know'. Do not attempt to guess or hallucinate."*
   - Generation temperature is set to `0.1` for maximum precision and predictability.

## 🚀 How to Run Locally

### 1. Backend Setup
Open a terminal and navigate to the project root:
```bash
# 1. Create a virtual environment and activate it
python -m venv venv

# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Add your OpenRouter API Key
# Open backend/.env and replace the placeholder with your OpenRouter key:
# OPENROUTER_API_KEY=your_openrouter_key_here

# 4. Ingest Documents (Build the Vector Index)
# (Ensure documents are downloaded and FAISS index builds)
python backend/ingest.py

# 5. Start the FastAPI Server
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
*The API will be available at `http://localhost:8000`. You can view interactive docs and test the query manually at `http://localhost:8000/docs`.*

### 2. Frontend Setup
Open a **new** terminal window and navigate to the project root:
```bash
cd frontend

# Install Vite and React dependencies
npm install

# Start the frontend dev server
npm run dev
```
*The UI will start (usually at `http://localhost:5173`). Open the link printed in your terminal to start chatting with the documents.*

## 🧪 Optional: Local LLM Integration (Bonus)

To swap OpenRouter for a local LLM, we can use Ollama. This takes processing fully offline.
1. Download and install [Ollama](https://ollama.com).
2. Pull a local model: `ollama run llama3:8b` (or `phi3` for a faster 3B model).
3. Update `backend/rag_core.py` initialization to point to Ollama's local, OpenAI-compatible endpoint:
   ```python
   self.client = OpenAI(
       base_url="http://localhost:11434/v1",
       api_key="ollama" # api_key is required by the client, but Ollama ignores it
   )
   self.model_name = "llama3:8b"
   ```

## 📊 Deployment Strategy

1. **Frontend**: Deploy the React App statics easily on **Vercel** or **Netlify**. Update `frontend/src/App.jsx` to point to the production backend URL instead of `localhost:8000`.
2. **Backend**: Host the Streamlit app on Streamlit Cloud. Commit the data/index FAISS folder to the repo (or configure Streamlit Cloud to run ingest.py as a pre‑deploy step). Set OPENROUTER_API_KEY as a secret/environment variable in the Streamlit app settings/dashboard.

