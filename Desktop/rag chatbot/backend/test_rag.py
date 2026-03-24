import os
import sys

# Add current dir to path just in case
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_core import RAGPipeline

def main():
    try:
        pipeline = RAGPipeline()
        query = "What is the main topic of the documents?"
        print(f"Querying: {query}")
        contexts = pipeline.retrieve(query, top_k=2)
        print("Contexts retrieved:")
        for c in contexts:
            print(f" - {c['source']} (score: {c['score']})")
        
        answer = pipeline.generate_answer(query, contexts)
        print(f"\nAnswer:\n{answer}")
        
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main()
