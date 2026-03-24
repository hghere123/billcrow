import os
import streamlit as st

from backend.rag_core import RAGPipeline  # adjust module name to your file


@st.cache_resource(show_spinner=True)
def load_rag():
    return RAGPipeline()


def main():
    st.set_page_config(page_title="Indecimal RAG Chatbot", page_icon="🏠")
    st.title("Indecimal AI Assistant")
    st.write("Ask anything about Indecimal’s pricing, process, or guarantees.")

    # Initialize pipeline once
    try:
        rag = load_rag()
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {e}")
        st.stop()

    # Simple chat UI
    if "history" not in st.session_state:
        st.session_state.history = []

    for role, text in st.session_state.history:
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Assistant:** {text}")

    question = st.text_input("Your question", placeholder="How does Indecimal reduce hidden surprises in pricing?")
    if st.button("Ask") and question.strip():
        st.session_state.history.append(("user", question))
        with st.spinner("Thinking..."):
            contexts = rag.retrieve(question, top_k=3)
            answer = rag.generate_answer(question, contexts)
        st.session_state.history.append(("assistant", answer))
        st.experimental_rerun()


if __name__ == "__main__":
    main()

