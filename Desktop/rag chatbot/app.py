import os
import streamlit as st

from backend.rag_core import RAGPipeline  # adjust module name to your file


@st.cache_resource(show_spinner=True)
def load_rag(chunk_size: int = 512, chunk_overlap: int = 128):
    """
    Create and cache the RAG pipeline.

    Assumes RAGPipeline can accept chunk_size and chunk_overlap.
    If not, remove these arguments and wire them however your class expects.
    """
    return RAGPipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def main():
    st.set_page_config(page_title="Indecimal RAG Chatbot", page_icon="🏠")
    st.title("Indecimal AI Assistant")
    st.write("Ask anything about Indecimal’s pricing, process, or guarantees.")

    # ---- Sidebar: chunking configuration ----
    st.sidebar.header("Chunking settings")

    chunk_size = st.sidebar.slider(
        "Chunk size",
        min_value=256,
        max_value=2048,
        value=512,
        step=128,
        help="Number of tokens/characters per chunk (depends on your implementation).",
    )

    chunk_overlap = st.sidebar.slider(
        "Chunk overlap",
        min_value=0,
        max_value=512,
        value=128,
        step=32,
        help="Overlap between consecutive chunks.",
    )

    st.sidebar.caption(
        f"Using chunk_size={chunk_size}, chunk_overlap={chunk_overlap}."
    )

    # Initialize pipeline once, parameterized by chunking
    try:
        rag = load_rag(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {e}")
        st.stop()

    # ---- Chat UI ----
    if "history" not in st.session_state:
        st.session_state.history = []

    # show previous messages
    for role, text in st.session_state.history:
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Assistant:** {text}")

    question = st.text_input(
        "Your question",
        placeholder="How does Indecimal reduce hidden surprises in pricing?",
    )

    if st.button("Ask") and question.strip():
        # add user message
        st.session_state.history.append(("user", question))

        # generate answer
        with st.spinner("Thinking..."):
            contexts = rag.retrieve(question, top_k=3)
            answer = rag.generate_answer(question, contexts)

        # add assistant message
        st.session_state.history.append(("assistant", answer))
        # no explicit rerun needed; Streamlit will rerun automatically


if __name__ == "__main__":
    main()
