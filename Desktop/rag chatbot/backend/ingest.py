import os
import glob
import json
import gdown
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Create data directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
INDEX_DIR = os.path.join(BASE_DIR, "data", "index")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Google Drive File IDs corresponding to the PDFs provided
DRIVE_FILES = {
    "doc1.pdf": "1oWcyH0XkzpHeWozMBWJSFEUEw70Lrc2-",
    "doc2.pdf": "1m1SudlRSlEK7y_-jweDjhPB5VVWzmQ7-",
    "doc3.pdf": "1suFO8EBLxRH6hKKcJln4a9PRsOGu2oYj"
}

def download_pdfs():
    print("Downloading PDFs from Google Drive...")
    for filename, file_id in DRIVE_FILES.items():
        output_path = os.path.join(RAW_DIR, filename)
        if not os.path.exists(output_path):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False)
            print(f"Downloaded {filename}")
        else:
            print(f"{filename} already exists. Skipping download.")

def extract_text_from_pdfs():
    print("Extracting text from PDFs (or fallback to text if plain text)...")
    docs = []
    pdf_files = glob.glob(os.path.join(RAW_DIR, "*.pdf"))
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            docs.append({"filename": filename, "text": text})
        except Exception as e:
            print(f"Could not parse {filename} as PDF, falling back to plain text. Error: {e}")
            with open(pdf_path, 'r', encoding='utf-8', errors='ignore') as f:
                docs.append({"filename": filename, "text": f.read()})
    return docs

def chunk_text(docs):
    print("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = []
    for doc in docs:
        splits = text_splitter.split_text(doc["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "source": doc["filename"],
                "chunk_id": f'{doc["filename"]}_chunk{i}',
                "text": split
            })
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def embed_and_store(chunks):
    print("Generating embeddings...")
    # Using a fast, local embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Extract text for embedding
    texts = [chunk["text"] for chunk in chunks]
    
    # Generate embeddings (this will return a numpy array)
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    # Create FAISS index (inner product for cosine similarity is used if vectors are normalized,
    # but L2 distance is perfectly fine and often default for generic embeddings).
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save the FAISS index
    index_path = os.path.join(INDEX_DIR, "vector_store.index")
    faiss.write_index(index, index_path)
    
    # Save the chunk metadata (so we can map back from index id to text)
    metadata_path = os.path.join(INDEX_DIR, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
        
    print(f"Index and metadata saved to {INDEX_DIR}")

if __name__ == "__main__":
    download_pdfs()
    docs = extract_text_from_pdfs()
    chunks = chunk_text(docs)
    embed_and_store(chunks)
    print("Ingestion complete!")
