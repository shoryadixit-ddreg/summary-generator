import streamlit as st
import requests
import json
from typing import Optional, Callable

# ---------- Configuration ----------

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="PDF Summary Generator",
    page_icon="📄",
    layout="wide",
)

# ---------- Session State Initialization ----------

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summary" not in st.session_state:
    st.session_state.summary = None

# ---------- Helper Functions ----------


def upload_pdf(file) -> Optional[dict]:
    """Upload PDF to the backend API."""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        # No timeout (unlimited) to handle large documents on CPU
        response = requests.post(f"{API_BASE_URL}/upload/", files=files, timeout=None)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the backend server. Make sure it's running.")
        return None
    except Exception as e:
        st.error(f"❌ Upload failed: {str(e)}")
        return None


def query_pdf(session_id: str, question: str, on_event: Callable[[dict], None]) -> bool:
    """Stream a query answer from the backend as SSE events. Returns True on success."""
    try:
        data = {"session_id": session_id, "question": question}
        with requests.post(
            f"{API_BASE_URL}/query/",
            data=data,
            stream=True,
            timeout=None,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                try:
                    payload = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                on_event(payload)
                if payload.get("type") in ("final", "error"):
                    return payload.get("type") == "final"
        return True
    except requests.exceptions.Timeout:
        on_event(
            {
                "type": "error",
                "detail": "Query timed out. The server may be busy. Please try again.",
            }
        )
        return False
    except Exception as e:
        on_event({"type": "error", "detail": f"Query failed: {str(e)}"})
        return False


def summarize_pdf(session_id: str, on_event: Callable[[dict], None]) -> bool:
    """Stream summary generation events from the backend. Returns True on success."""
    try:
        with requests.post(
            f"{API_BASE_URL}/summarize/",
            data={"session_id": session_id},
            stream=True,
            timeout=None,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                try:
                    payload = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                on_event(payload)
                if payload.get("type") == "final":
                    return True
                if payload.get("type") == "error":
                    return False
        return True
    except requests.exceptions.Timeout:
        on_event(
            {
                "type": "error",
                "detail": "Summary generation timed out. The document may be too large or the server is running on CPU. Try a smaller document or wait longer.",
            }
        )
        return False
    except Exception as e:
        on_event({"type": "error", "detail": f"Summary generation failed: {str(e)}"})
        return False


def check_api_health() -> bool:
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False


# ---------- UI ----------

st.title("📄 PDF Summary Generator")
st.markdown(
    "Upload a PDF document, then ask questions or generate a summary "
    "using **LangChain** + **HuggingFace**."
)

# Sidebar - API Status & Upload
with st.sidebar:
    st.header("📡 API Status")
    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ Backend is running")
    else:
        st.error("❌ Backend is not available")
        st.info(
            "Run the backend with: `python -m uvicorn app:app --host 0.0.0.0 --port 8000`"
        )
        st.stop()

    st.divider()

    st.header("📤 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file", type=["pdf"], accept_multiple_files=False
    )

    if uploaded_file is not None:
        with st.spinner("Processing PDF... (this may take a while on first run)"):
            result = upload_pdf(uploaded_file)

        if result:
            st.session_state.session_id = result["session_id"]
            st.session_state.filename = result["filename"]
            st.session_state.total_chunks = result["total_chunks"]
            st.session_state.chat_history = []
            st.session_state.summary = None
            st.success(f"✅ {result['filename']} processed!")
            st.info(f"📊 Chunks: {result['total_chunks']}")
            st.info(f"🆔 Session: `{result['session_id'][:12]}...`")

    st.divider()

    if st.session_state.session_id:
        st.header("📋 Session Info")
        st.write(f"**File:** {st.session_state.filename}")
        st.write(f"**Chunks:** {st.session_state.total_chunks}")
        st.write(f"**Session ID:** `{st.session_state.session_id[:16]}...`")

        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.session_id = None
            st.session_state.filename = None
            st.session_state.total_chunks = None
            st.session_state.chat_history = []
            st.session_state.summary = None
            st.rerun()

# Main Area - Tabs
if st.session_state.session_id is None:
    # No document loaded - show welcome screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👈 Upload a PDF from the sidebar to get started.")
        st.markdown("""
        ### How it works:
        1. **Upload** a PDF document from the sidebar
        2. **Ask questions** about the document content
        3. **Generate a summary** of the entire document

        The system uses:
        - 🧠 **HuggingFace** `all-MiniLM-L6-v2` for embeddings
        - 🤖 **HuggingFace** `mistralai/Mistral-7B-Instruct-v0.1` for text generation
        - 📚 **LangChain** for document processing & RAG
        - ⚡ **FAISS** for vector similarity search
        """)

else:
    # Document loaded - show tabs
    tab1, tab2, tab3 = st.tabs(
        ["💬 Ask Questions", "📝 Summary", "ℹ️ About the Document"]
    )

    # -------- TAB 1: Q&A --------
    with tab1:
        st.subheader(f"Ask questions about **{st.session_state.filename}**")

        # Chat history display
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Type your question about the PDF..."):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get assistant response with live token streaming
            with st.chat_message("assistant"):
                answer_placeholder = st.empty()
                source_placeholder = st.empty()
                answer_tokens: list[str] = []

                def handle_query_event(payload: dict):
                    event_type = payload.get("type")
                    if event_type == "progress":
                        status = payload.get("status", "thinking")
                        answer_placeholder.info(f"⏳ {status.capitalize()}...")
                    elif event_type == "token":
                        answer_tokens.append(payload.get("text", ""))
                        answer_placeholder.markdown("".join(answer_tokens))
                    elif event_type == "final":
                        final_answer = payload.get("answer", "No answer returned")
                        answer_placeholder.markdown(final_answer)
                        source_count = payload.get("total_context_chunks", 0)
                        if source_count:
                            source_placeholder.caption(
                                f"🔍 Based on {source_count} relevant context chunks (document + regulatory guidance)"
                            )
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": final_answer}
                        )
                    elif event_type == "error":
                        detail = payload.get("detail", "Failed to get an answer.")
                        answer_placeholder.error(f"❌ {detail}")

                query_pdf(st.session_state.session_id, prompt, handle_query_event)

        # Clear chat button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

    # -------- TAB 2: Summary --------
    with tab2:
        st.subheader(f"Summary of **{st.session_state.filename}**")

        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(
                "📝 Generate Summary", type="primary", use_container_width=True
            ):
                progress_bar = st.progress(0, text="Preparing...")
                status = st.empty()
                section_count = st.empty()

                def handle_event(payload: dict):
                    event_type = payload.get("type")
                    if event_type == "start":
                        status.info("Starting summary generation...")
                    elif event_type == "progress":
                        total = payload.get("total") or 1
                        done = payload.get("done") or 0
                        progress_bar.progress(
                            min(done / total, 1.0),
                            text=f"Summarizing chunks {done}/{total}...",
                        )
                    elif event_type == "summary_part":
                        idx = payload.get("index", 0) + 1
                        total = payload.get("total", "...")
                        section_count.caption(
                            f"📄 {idx}/{total} section summaries generated — merging..."
                        )
                    elif event_type == "reduce":
                        status.info(
                            f"🔄 Merging summaries — stage {payload.get('level')}, "
                            f"{payload.get('count')} sections remaining..."
                        )
                    elif event_type == "final":
                        st.session_state.summary = payload["summary"]
                        progress_bar.progress(1.0, text="Done")
                        status.success("✅ Summary generated!")
                        section_count.empty()
                    elif event_type == "error":
                        status.error(f"❌ {payload.get('detail', 'Summary generation failed.')}")

                summarize_pdf(st.session_state.session_id, handle_event)

        with col2:
            if st.session_state.summary:
                st.download_button(
                    label="💾 Download Summary",
                    data=st.session_state.summary,
                    file_name=f"{st.session_state.filename}_summary.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

        if st.session_state.summary:
            st.divider()
            st.markdown(st.session_state.summary)
        else:
            st.info("Click the button above to generate a summary.")

    # -------- TAB 3: About --------
    with tab3:
        st.subheader("Document Information")
        st.write(f"**Filename:** {st.session_state.filename}")
        st.write(f"**Total Chunks:** {st.session_state.total_chunks}")
        st.write(f"**Session ID:** `{st.session_state.session_id}`")

        st.divider()
        st.subheader("How it works")
        st.markdown("""
        ### Pipeline:
        1. **PDF Upload** → Text extracted via PyPDFLoader
        2. **Text Splitting** → Document split into chunks (2000 chars, 100 overlap)
        3. **Embedding** → Chunks embedded using HuggingFace `all-MiniLM-L6-v2`
        4. **Vector Store** → Embeddings stored in FAISS index for similarity search
        5. **Retrieval** → Top-5 relevant chunks retrieved for each query
        6. **Generation** → LLM (`mistralai/Mistral-7B-Instruct-v0.1`) generates answer from retrieved context

        ### Tech Stack:
        - **Backend:** FastAPI + LangChain + HuggingFace + FAISS
        - **Frontend:** Streamlit
        - **Models:** `all-MiniLM-L6-v2` (embeddings), `mistralai/Mistral-7B-Instruct-v0.1` (LLM)
        """)
