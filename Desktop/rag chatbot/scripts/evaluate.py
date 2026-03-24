import requests
import json

TEST_QUESTIONS = [
    "What is the main topic of the first document?",
    "What is the overarching goal discussed?",
    "Who is the target audience for these manuals?",
    "What are the key technical requirements?",
    "How is the pricing or support model structured?",
    "What happens if a user forgets their password?",
    "What security protocols are in place?",
    "How do I contact the sales team?"
]

API_URL = "http://localhost:8000/rag/query"

def run_evaluation():
    print("==========================================")
    print("   Mini RAG Pipeline Evaluation Script    ")
    print("==========================================\n")
    
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"Q{i}: {q}")
        try:
            response = requests.post(
                API_URL, 
                json={"question": q},
                timeout=30 # Give LLM time to generate
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"Answer: {data.get('answer', 'No answer')[:200]}...\n")
                print("Retrieved Sources:")
                for idx, ctx in enumerate(data.get('contexts', [])):
                    print(f"  [{idx+1}] Source: {ctx['source']} (Relevance Score: {ctx['score']:.3f})")
                
                # --- MANUAL EVALUATION PROMPT ---
                # To extend this into an interactive benchmark:
                # rating = input("Rate relevance/groundedness (1-5): ")
                # log_results(q, data, rating)
                
            else:
                print(f"[Error] API returned {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            print("[Error] Failed to connect to FastAPI backend. Is it running on port 8000?")
            break
        except Exception as e:
            print(f"[Error] {e}")
        print("-" * 50)

if __name__ == "__main__":
    run_evaluation()
