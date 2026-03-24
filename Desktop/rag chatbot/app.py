import os
import streamlit as st
from backend.rag_core import RAGPipeline


@st.cache_resource(show_spinner=True)
def load_rag():
    return RAGPipeline()


def main():
    st.set_page_config(page_title="Indecimal RAG Chatbot", page_icon="🏗️", layout="wide")
    st.title("Indecimal AI Assistant")
    st.write("Ask anything about Indecimal’s pricing, process, or guarantees.")

    # Sidebar chunking settings (UI only for now)
    st.sidebar.header("Document & Chunking settings")

    chunk_size = st.sidebar.slider("Chunk size", 256, 2048, 512, 128)
    chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 512, 128, 32)
    top_k = st.sidebar.slider("Top‑k retrieved chunks", 1, 10, 3, 1)

    st.sidebar.caption(
        f"Using chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, top_k={top_k}."
    )

    try:
        rag = load_rag()
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {e}")
        st.stop()

    if "history" not in st.session_state:
        st.session_state.history = []

    st.subheader("Chat")

    for role, text in st.session_state.history:
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Assistant:** {text}")

    question = st.text_input(
        "Your question",
        placeholder="What factors affect construction project delays?",
    )

    if st.button("Ask") and question.strip():
        st.session_state.history.append(("user", question))

        with st.spinner("Retrieving and generating answer..."):
            contexts = rag.retrieve(question, top_k=top_k)
            answer = rag.generate_answer(question, contexts)

        st.session_state.history.append(("assistant", answer))

        st.markdown("### Retrieved context")
        if not contexts:
            st.info("No context retrieved.")
        else:
            for i, c in enumerate(contexts, start=1):
                st.markdown(f"**Chunk {i}:**")
                st.write(c)

        st.markdown("### Final answer")
        st.write(answer)


if __name__ == "__main__":
    main()
