import streamlit as st
import numpy as np
import os
import sys
from PIL import Image
import tempfile

# Set ALL environment variables BEFORE ANYTHING ELSE
script_dir = os.path.dirname(os.path.abspath(__file__))
easyocr_dir = os.path.join(script_dir, "easyocr_models")
os.makedirs(easyocr_dir, exist_ok=True)
os.environ["EASYOCR_MODULE_PATH"] = easyocr_dir
os.environ["HOME"] = script_dir
os.environ["USERPROFILE"] = script_dir
os.environ["PYTHONIOENCODING"] = "utf-8"

st.set_page_config(page_title="Local Multilingual RAG", layout="wide", page_icon="📄")

# Clear cache button
if st.button("🔄 Clear All Cache"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.success("Cache cleared! Please refresh the page.")

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stTextArea textarea { font-family: monospace; font-size: 13px; }
    .log-box {
        background: #1a1d27; border-left: 3px solid #00d4aa;
        padding: 10px 14px; border-radius: 4px;
        font-family: monospace; font-size: 12px;
        color: #c8fae8; margin: 4px 0;
    }
    .metric-box {
        background: #1a1d27; border: 1px solid #2a2d3a;
        padding: 12px; border-radius: 8px; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("📄 Local OCR & Dynamic RAG System")
st.caption("Bangla + English | Fully Local | No External APIs")
st.markdown("---")

# ── Sidebar: Metadata Filters ────────────────────────────────────────────────
st.sidebar.header("⚙️ Manual Metadata Filters")
st.sidebar.markdown("Enable filters to narrow the search scope.")

doc_date   = st.sidebar.date_input("📅 Document Date")
doc_type   = st.sidebar.selectbox("📁 Document Type", ["Official", "Personal", "Invoice", "Letter", "Academic", "News"])
doc_lang   = st.sidebar.radio("🌐 Document Language", ["Bangla", "English", "Mixed"])

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Active Filters")
use_date_filter = st.sidebar.checkbox("Filter by Date",     value=False)
use_type_filter = st.sidebar.checkbox("Filter by Doc Type", value=False)
use_lang_filter = st.sidebar.checkbox("Filter by Language", value=True)

# ── File Upload ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📤 Upload Scanned Document (PDF / JPG / PNG)",
    type=["png", "jpg", "jpeg", "pdf"]
)

# ── Cached model loaders ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False, ttl=0, max_entries=1)
def load_ocr_v4():
    import easyocr
    return easyocr.Reader(
        ["bn", "en"], 
        gpu=False, 
        verbose=False, 
        model_storage_directory=easyocr_dir,
        user_network_directory=easyocr_dir
    )

@st.cache_resource(show_spinner=False)
def load_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

# ── Helpers ───────────────────────────────────────────────────────────────────
def log(msg: str):
    st.markdown(f'<div class="log-box">▶ {msg}</div>', unsafe_allow_html=True)

def pdf_to_images(file_bytes: bytes):
    import pdf2image
    return pdf2image.convert_from_bytes(file_bytes)

def ocr_image(reader, img) -> str:
    arr = np.array(img)
    results = reader.readtext(arr, detail=1)
    lines, confidences = [], []
    for (_, text, conf) in results:
        lines.append(text)
        confidences.append(conf)
    avg_conf = round(sum(confidences) / len(confidences) * 100, 1) if confidences else 0
    return " ".join(lines), avg_conf

# ── Chroma helpers (no .persist() — auto-persists in newer chromadb) ─────────
CHROMA_DIR = "./chroma_db"

def get_vectorstore(embeddings):
    from langchain_community.vectorstores import Chroma
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

def add_to_vectorstore(chunks, metadatas, embeddings):
    from langchain_community.vectorstores import Chroma
    Chroma.from_texts(
        chunks,
        embeddings,
        metadatas=metadatas,
        persist_directory=CHROMA_DIR,
    )

# ── Main flow ─────────────────────────────────────────────────────────────────
if uploaded_file:
    st.success(f"✅ Uploaded: **{uploaded_file.name}**  ({uploaded_file.size/1024:.1f} KB)")

    # Load models
    with st.spinner("⏳ Loading OCR model (first run downloads ~150 MB)…"):
        try:
            reader = load_ocr_v4()
            log("EasyOCR loaded — languages: Bangla (bn) + English (en)")
        except Exception as e:
            st.error(f"❌ Error loading OCR: {e}")
            import traceback
            st.error(traceback.format_exc())
            st.stop()

    with st.spinner("⏳ Loading embedding model…"):
        try:
            embeddings = load_embeddings()
            log("Embedding model loaded — paraphrase-multilingual-MiniLM-L12-v2")
        except Exception as e:
            st.error(f"❌ Error loading embeddings: {e}")
            st.stop()

    file_bytes = uploaded_file.read()
    is_pdf     = uploaded_file.type == "application/pdf"

    # Convert to images
    with st.spinner("📄 Converting document…"):
        if is_pdf:
            images = pdf_to_images(file_bytes)
            log(f"PDF converted → {len(images)} page(s)")
        else:
            import io
            images = [Image.open(io.BytesIO(file_bytes))]
            log("Image loaded successfully")

    # Show preview + OCR side-by-side
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📄 Document Preview")
        st.image(images[0], caption="Page 1", use_container_width=True)

    with col2:
        st.subheader("📝 OCR Processing Logs")
        full_text    = ""
        total_conf   = []
        char_counts  = []

        for i, img in enumerate(images):
            with st.spinner(f"🔍 OCR on page {i+1}/{len(images)}…"):
                page_text, conf = ocr_image(reader, img)
                full_text      += page_text + "\n\n"
                total_conf.append(conf)
                char_counts.append(len(page_text))
                log(f"Page {i+1}: {len(page_text)} chars | confidence {conf}%")

        avg_conf = round(sum(total_conf) / len(total_conf), 1) if total_conf else 0

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Characters", f"{sum(char_counts):,}")
        m2.metric("Pages Processed",  len(images))
        m3.metric("Avg OCR Confidence", f"{avg_conf}%")

        st.text_area("Extracted Text", full_text.strip(), height=250)

    # Chunk + embed + store
    st.markdown("---")
    with st.spinner("🧩 Chunking and indexing into ChromaDB…"):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks   = splitter.split_text(full_text.strip())

        if not chunks:
            st.warning("⚠️ No text extracted. Try a clearer image.")
            st.stop()

        metadatas = [{
            "date":     str(doc_date),
            "type":     doc_type,
            "lang":     doc_lang,
            "filename": uploaded_file.name,
        } for _ in chunks]

        add_to_vectorstore(chunks, metadatas, embeddings)

        log(f"Chunked into {len(chunks)} pieces (size=500, overlap=50)")
        log(f"Metadata saved → date:{doc_date} | type:{doc_type} | lang:{doc_lang}")
        log(f"Stored in ChromaDB at {CHROMA_DIR}")
        st.success(f"✅ Indexed **{len(chunks)} chunks** into vector store!")

    # ── RAG Query ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💬 Ask a Question (Bangla বা English)")

    # Show active filters
    active = []
    if use_date_filter: active.append(f"Date = {doc_date}")
    if use_type_filter: active.append(f"Type = {doc_type}")
    if use_lang_filter: active.append(f"Language = {doc_lang}")
    if active:
        st.info("🔍 Active filters: " + " | ".join(active))
    else:
        st.info("🔍 No filters active — searching all documents")

    query = st.text_input("Your question:", placeholder="e.g. এই নথিতে কী বলা আছে? / What does this document say?")

    if query:
        with st.spinner("🤖 Searching and generating answer…"):
            try:
                from langchain_community.vectorstores import Chroma
                from langchain_community.llms import Ollama
                from langchain_core.prompts import PromptTemplate
                from langchain.chains.combine_documents import create_stuff_documents_chain
                from langchain.chains.retrieval import create_retrieval_chain

                vectorstore = get_vectorstore(embeddings)

                # Build filter dict dynamically — only enabled filters
                filter_dict = {}
                if use_date_filter: filter_dict["date"] = str(doc_date)
                if use_type_filter: filter_dict["type"] = doc_type
                if use_lang_filter: filter_dict["lang"] = doc_lang

                search_kwargs = {"k": 4}
                if filter_dict:
                    search_kwargs["filter"] = filter_dict

                log(f"Vector search with filter: {filter_dict if filter_dict else 'none'}")

                retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

                # BM25 hybrid search
                try:
                    from langchain_community.retrievers import BM25Retriever
                    from langchain.retrievers import EnsembleRetriever

                    bm25 = BM25Retriever.from_texts(chunks, metadatas=metadatas)
                    bm25.k = 4
                    hybrid_retriever = EnsembleRetriever(
                        retrievers=[bm25, retriever],
                        weights=[0.3, 0.7]
                    )
                    log("Hybrid search: BM25 (30%) + Vector (70%)")
                    active_retriever = hybrid_retriever
                except Exception:
                    log("BM25 unavailable — using vector-only search")
                    active_retriever = retriever

                prompt = PromptTemplate.from_template("""
You are a helpful assistant. Answer the question based ONLY on the context below.
If the context is in Bangla, answer in Bangla. If English, answer in English.
If you cannot find the answer in the context, say so honestly.

Context:
{context}

Question: {input}

Answer:""")

                llm = Ollama(model="llama3")
                doc_chain      = create_stuff_documents_chain(llm, prompt)
                retrieval_chain = create_retrieval_chain(active_retriever, doc_chain)
                response       = retrieval_chain.invoke({"input": query})

                st.subheader("✅ Answer")
                st.markdown(f"> {response['answer']}")

                with st.expander("📚 Source Chunks Used"):
                    for i, doc in enumerate(response.get("context", [])):
                        st.markdown(f"**Chunk {i+1}**")
                        st.write(doc.page_content)
                        st.caption(f"Metadata: {doc.metadata}")
                        st.markdown("---")

            except Exception as e:
                err = str(e)
                if "ollama" in err.lower() or "connection" in err.lower():
                    st.error("❌ Ollama is not running!")
                    st.code("ollama pull llama3\nollama run llama3", language="bash")
                    st.info("Run the above commands in a separate terminal, then try again.")
                else:
                    st.error(f"❌ RAG Error: {e}")

else:
    # Landing state
    st.markdown("""
    ### How to use
    1. **Set metadata** in the left sidebar (date, type, language)
    2. **Upload** a scanned image or PDF (Bangla / English / Mixed)
    3. **Watch** OCR extract text locally — no internet needed
    4. **Ask** any question about the document in Bangla or English
    5. **Toggle filters** in the sidebar to narrow the search
    """)
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.info("🔒 **100% Local**\nNo data leaves your machine")
    c2.info("🌐 **Bilingual**\nBangla + English OCR")
    c3.info("🔍 **Hybrid Search**\nBM25 + Vector similarity")