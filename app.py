import os
import json
import streamlit as st
from rag_engine import (
    process_uploaded_files,
    process_sample_file,
    query_rag,
    evaluate_retrieval,
)

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")

st.title("📄 Grounded RAG Document Q&A")
st.caption("Ask questions across your PDFs & notes. Answers are strictly grounded in retrieved FAISS chunks with inline citations.")

# Support optional Streamlit Cloud secret so live visitors can test immediately
default_key = ""
try:
    default_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    default_key = os.getenv("GEMINI_API_KEY", "")

# --- Sidebar: Configuration & Ingestion ---
with st.sidebar:
    st.header("1. Configuration")
    api_key = st.text_input(
        "Gemini API Key",
        value=default_key,
        type="password",
        placeholder="AIza...",
        help="Pre-filled if configured in Streamlit Secrets, or paste your own key."
    )
    model_name = st.text_input("Gemini Model", value="gemini-3.8-flash")
    
    with st.expander("RAG Hyperparameters", expanded=False):
        chunk_size = st.slider("Chunk Size (chars)", 300, 1500, 500, step=100)
        chunk_overlap = st.slider("Chunk Overlap (chars)", 0, 400, 100, step=50)
        top_k = st.slider("Top-K Retrieved Chunks", 1, 8, 4)

    st.header("2. Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    if st.button("Build Vector Index", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please enter your Gemini API key first.")
        elif not uploaded_files:
            st.error("Please upload at least one document.")
        else:
            with st.spinner("Chunking, embedding & indexing in FAISS..."):
                vs, chunks = process_uploaded_files(
                    uploaded_files, api_key.strip(), chunk_size, chunk_overlap
                )
                st.session_state["vector_store"] = vs
                st.session_state["chunk_count"] = len(chunks)
                st.session_state["active_source"] = f"{len(uploaded_files)} uploaded file(s)"
                st.session_state["messages"] = []
            st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s)!")

    st.markdown("---")
    st.caption("No document handy? Try the bundled sample:")
    if st.button("Load Sample Document", use_container_width=True):
        sample_path = os.path.join(os.path.dirname(__file__), "sample_doc.md")
        if not api_key:
            st.error("Please enter your Gemini API key first.")
        elif not os.path.exists(sample_path):
            st.error("sample_doc.md not found in project directory.")
        else:
            with st.spinner("Indexing bundled sample document..."):
                vs, chunks = process_sample_file(
                    sample_path, api_key.strip(), chunk_size, chunk_overlap
                )
                st.session_state["vector_store"] = vs
                st.session_state["chunk_count"] = len(chunks)
                st.session_state["active_source"] = "sample_doc.md"
                st.session_state["messages"] = []
            st.success(f"Indexed {len(chunks)} chunks from sample_doc.md!")

    if "chunk_count" in st.session_state:
        st.info(
            f"**Active Index:** {st.session_state['chunk_count']} chunks "
            f"({st.session_state.get('active_source', 'Ready')})"
        )

# --- Main Tabs ---
tab_chat, tab_eval = st.tabs(["💬 Document Q&A", "📊 Retrieval Evaluation (Hit Rate)"])

with tab_chat:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("View Retrieved Source Passages"):
                    for src in msg["sources"]:
                        st.markdown(
                            f"**[{src['id']}] {src['source']} — Page {src['page']}** "
                            f"*(L2 Distance: {src['score']})*"
                        )
                        st.info(src["content"])

    if prompt := st.chat_input("Ask a question about your uploaded documents..."):
        if "vector_store" not in st.session_state:
            st.warning("Upload documents or click **Load Sample Document** in the sidebar first.")
        elif not api_key:
            st.warning("Please enter your Gemini API key in the sidebar.")
        else:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving passages and generating cited answer..."):
                    result = query_rag(
                        st.session_state["vector_store"],
                        prompt,
                        api_key.strip(),
                        model_name=model_name,
                        top_k=top_k
                    )
                    st.markdown(result["answer"])
                    with st.expander("View Retrieved Source Passages"):
                        for src in result["sources"]:
                            st.markdown(
                                f"**[{src['id']}] {src['source']} — Page {src['page']}** "
                                f"*(L2 Distance: {src['score']})*"
                            )
                            st.info(src["content"])

            st.session_state["messages"].append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"]
            })

with tab_eval:
    st.subheader("Measure Retrieval Quality (Hit Rate@K & MRR)")
    st.markdown(
        "Test whether your chunking and embedding pipeline actually retrieves the right passages. "
        "Provide a JSON list of test questions and a ground-truth phrase that must appear in the top-K chunks."
    )

    sample_eval = json.dumps([
        {"question": "What university and GPA are listed?", "expected_substring": "University of Maryland, Baltimore County"},
        {"question": "What message broker was used in the JPMorganChase simulation?", "expected_substring": "Kafka"},
        {"question": "How many concurrent requests per second did the flash-sale engine handle?", "expected_substring": "500"},
        {"question": "What differential privacy mechanism was used in the smart home pipeline?", "expected_substring": "Laplace"},
        {"question": "What WebRTC server and object detection model were used in the video meeting project?", "expected_substring": "YOLO"},
        {"question": "What course is graded as a Graduate Assistant?", "expected_substring": "Discrete Structures"},
        {"question": "What hand landmark detector was used in the sign language project?", "expected_substring": "MediaPipe"}
    ], indent=2)

    eval_input = st.text_area("Evaluation Dataset (JSON)", value=sample_eval, height=230)

    if st.button("Run Retrieval Benchmark"):
        if "vector_store" not in st.session_state:
            st.warning("Build the vector index first before running evaluation.")
        else:
            try:
                eval_cases = json.loads(eval_input)
                metrics = evaluate_retrieval(
                    st.session_state["vector_store"], eval_cases, top_k=top_k
                )
                
                col1, col2, col3 = st.columns(3)
                col1.metric(f"Hit Rate@{top_k}", f"{metrics['hit_rate']}%")
                col2.metric("Mean Reciprocal Rank (MRR)", metrics["mrr"])
                col3.metric("Passed Queries", f"{metrics['hits']} / {metrics['total']}")
                
                st.dataframe(metrics["details"], use_container_width=True)
            except json.JSONDecodeError:
                st.error("Invalid JSON format. Please check your syntax.")