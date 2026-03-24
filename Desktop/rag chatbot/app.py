import os
import streamlit as st

from backend.rag_core import RAGPipeline  # adjust if your module path is different


# --------- Load RAG pipeline (cached) ---------
@st.cache_resource(show_spinner=True)
def load_rag(chunk_size: int, chunk_overlap: int):
    """
    Create and cache the RAG pipeline.

    Make sure your RAGPipeline __init__ accepts chunk_size and chunk_overlap.
    If it does not, remove these arguments here and wire them in rag_core instead.
    """
    return RAGPipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


# --------- Streamlit app ---------
def main():
    st.set_page_config(
        page_title="Indecimal RAG Chatbot",
        page_icon="🏗️",
        layout="wide",
    )

    st.title("Indecimal AI Assistant")
    st.write("Ask anything about Indecimal’s pricing, process, or guarantees.")

    # ---- Sidebar: document + chunking controls ----
    st.sidebar.header("Document & Chunking settings")

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

    top_k = st.sidebar.slider(
        "Top‑k retrieved chunks",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="How many chunks to use as context for each answer.",
    )

    st.sidebar.caption(
        f"Using chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, top_k={top_k}."
    )

    # ---- Initialize RAG pipeline ----
    try:
        rag = load_rag(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except TypeError as e:
        st.error(
            "RAGPipeline.__init__ does not accept chunk_size/chunk_overlap. "
            "Either add these parameters to RAGPipeline or remove them from load_rag."
        )
        st.exception(e)
        st.stop()
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {e}")
        st.stop()

    # ---- Chat history ----
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

    # ---- Ask button ----
    if st.button("Ask") and question.strip():
        st.session_state.history.append(("user", question))

        with st.spinner("Retrieving and generating answer..."):
            try:
                # retrieve top‑k chunks
                contexts = rag.retrieve(question, top_k=top_k)

                # generate grounded answer
                answer = rag.generate_answer(question, contexts)
            except Exception as e:
                st.error("Error while running the RAG pipeline.")
                st.exception(e)
                return

        st.session_state.history.append(("assistant", answer))

        # ---- Transparency: show retrieved chunks ----
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
