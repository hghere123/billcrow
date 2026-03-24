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
        print("Contexts retrieved. Now generating answer directly...")
        
        context_str = "\n\n---\n\n".join(
            [f"Source: {c['source']}\n{c['text']}" for c in contexts]
        )
        prompt = f"""Read the CONTEXT carefully and find the exact sentence that answers the QUESTION. 
You must output ONLY that exact sentence. Do not change any words. Do not explain.

CONTEXT:
{context_str}

QUESTION: {query}
EXACT SENTENCE ANSWER:"""

        print(f"Testing models...")
        for model in pipeline.model_names:
            print(f"Trying model: {model}")
            try:
                response = pipeline.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a strict document extraction assistant. You only answer based on the provided text."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                answer = response.choices[0].message.content
                print(f"Model {model} returned: {repr(answer)}")
                if answer:
                    print("SUCCESS: We found a model that gives an answer!")
                    break
            except Exception as e:
                print(f"Model {model} failed with error: {e}")
                
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main()
