import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)

INDEX_DIR = os.path.join(BASE_DIR, "data", "index")


class RAGPipeline:
    def __init__(self):
        print("Loading local embedding model...")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        print("Loading FAISS index...")
        index_path = os.path.join(INDEX_DIR, "vector_store.index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Missing FAISS index at {index_path}. Run ingest.py first."
            )
        self.index = faiss.read_index(index_path)

        print("Loading metadata...")
        metadata_path = os.path.join(INDEX_DIR, "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        load_dotenv(env_path, override=True)
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not self.api_key or self.api_key == "your_openrouter_key_here":
            print(
                "WARNING: OpenRouter API Key is missing. The API will return an error message to the user."
            )

        from openai import OpenAI

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=(
                self.api_key
                if self.api_key and self.api_key != "your_openrouter_key_here"
                else "dummy_key"
            ),
        )

        # Note: correct prefix is 'mistralai', not 'mistral-ai'
        self.model_names = [
            "mistralai/mistral-7b-instruct:free",
            "meta-llama/llama-2-7b-chat:free",
            "google/flan-t5-xxl:free",
            "openrouter/auto",
        ]

    # -------- Retrieval --------
    def retrieve(self, query: str, top_k: int = 3):
        query_embedding = self.encoder.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.metadata):
                chunk_data = self.metadata[idx]
                results.append(
                    {
                        "text": chunk_data["text"],
                        "source": chunk_data["source"],
                        "score": float(distances[0][i]),
                    }
                )
        return results

    # -------- FAQ first (zero hallucination) --------
    def _answer_from_faq_pattern(self, query: str, contexts: list):
        """
        If a context contains an explicit Q/A block where the Q is similar
        to the user query, return the A directly (no model, no hallucination).
        """
        q_norm = query.strip().lower()
        for c in contexts:
            text = c["text"]
            lines = text.splitlines()
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                if line_stripped.lower().startswith("q:"):
                    q_text = line_stripped[2:].strip().rstrip("?").lower()
                    # simple fuzzy-ish check
                    if q_text in q_norm or q_norm in q_text:
                        # look for next line starting with A:
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            if next_line.lower().startswith("a:"):
                                ans = next_line[2:].strip()
                                if ans:
                                    return ans
        return None

    # -------- Generation with safe reasoning --------
    def generate_answer(self, query: str, contexts: list):
        if not contexts:
            return "I don't know."

        # 1) Try FAQ pattern first (your Indecimal FAQ is in this format). [file:3]
        faq_answer = self._answer_from_faq_pattern(query, contexts)
        if faq_answer:
            return faq_answer

        # 2) If no direct FAQ hit, ask the model to reason, but in a grounded way.
        context_blocks = []
        for i, c in enumerate(contexts):
            context_blocks.append(f"[{i}] Source: {c['source']}\n{c['text']}")
        context_str = "\n\n---\n\n".join(context_blocks)

        prompt = f"""You are a careful assistant that must answer questions ONLY using the CONTEXT.

First, silently think about which 1–3 sentences in the CONTEXT are most relevant to the QUESTION.
Then answer the QUESTION in at most 2 short sentences, strictly based on those sentences.
If the CONTEXT does not contain enough information, answer exactly: "I don't know."

Do NOT mention the steps, your thinking, or the word CONTEXT in the final answer.
Do NOT invent facts that are not supported by the text.

CONTEXT:
{context_str}

QUESTION: {query}

Final answer:"""

        if not self.api_key or self.api_key == "your_openrouter_key_here":
            return (
                "ERROR: You requested the fast online model, but your OpenRouter API key is "
                "missing! Please create a free key at https://openrouter.ai/keys and paste it into `backend/.env`"
            )

        last_error = ""
        for model in self.model_names:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a strict document-grounded assistant. "
                                "Never use outside knowledge; if the answer is not clearly in the text, say \"I don't know.\""
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=128,
                )
                answer = response.choices[0].message.content.strip()

                # Final guardrail: if model says something very generic,
                # or contains no overlap with context, you can optionally add checks here.
                if not answer:
                    last_error = "Empty response"
                    continue
                return answer

            except Exception as e:
                err_str = str(e)
                if "401" in err_str:
                    return (
                        "API Error (401 Unauthorized): Your OpenRouter API key in backend/.env is invalid or rejected."
                    )
                last_error = err_str
                continue

        return (
            f"ERROR: All free AI models we tried failed or are offline. Last error: {last_error}"
        )
