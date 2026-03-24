import streamlit as st
from backend.rag_pipeline import RAGPipeline  # adjust file name if needed


@st.cache_resource(show_spinner=True)
def load_rag():
    # Load once and reuse across reruns
    return RAGPipeline()


def main():
    st.set_page_config(page_title="Indecimal RAG Chatbot", page_icon="🏠")
    st.title("Indecimal AI Assistant")
    st.write("Ask anything about Indecimal’s pricing, quality, or process, based only on the docs.")

    # Load pipeline
    try:
        rag = load_rag()
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {e}")
        st.stop()

    # Simple chat history
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
        placeholder="How does Indecimal reduce hidden surprises in pricing?",
    )

    if st.button("Ask") and question.strip():
        st.session_state.history.append(("user", question))
        with st.spinner("Thinking..."):
            contexts = rag.retrieve(question, top_k=3)
            answer = rag.generate_answer(question, contexts)
        st.session_state.history.append(("assistant", answer))
        st.experimental_rerun()

    # Optional: show retrieved sources
    with st.expander("View sources for last answer"):
        if st.session_state.get("history"):
            # last question + answer pair
            last_user = None
            last_answer = None
            for role, text in reversed(st.session_state.history):
                if role == "assistant" and last_answer is None:
                    last_answer = text
                elif role == "user" and last_user is None:
                    last_user = text
                    break

            if last_user:
                ctxs = rag.retrieve(last_user, top_k=3)
                for i, c in enumerate(ctxs):
                    st.markdown(f"**Source {i+1}: {c['source']}**")
                    st.write(c["text"])
        else:
            st.write("Ask a question first to see sources.")


if __name__ == "__main__":
    main()
