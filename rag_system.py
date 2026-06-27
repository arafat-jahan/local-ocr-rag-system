import streamlit as st
import numpy as np
import os
from PIL import Image
import shutil
from collections import defaultdict

# Set page config FIRST!
st.set_page_config(page_title="Local Multilingual RAG", layout="wide")

# App title and basic UI
st.title("📄 Local OCR & Dynamic RAG System")
st.markdown("---")

# Sidebar: Manual Metadata Filters
st.sidebar.header("⚙️ Manual Metadata Filters")
st.sidebar.markdown("---")
doc_date = st.sidebar.date_input("Document Date")
doc_type = st.sidebar.selectbox("Document Type", ["Official", "Personal", "Invoice", "Letter"])
doc_lang = st.sidebar.radio("Document Language", ["Bangla", "English", "Mixed"])

uploaded_file = st.file_uploader("📤 Upload Scanned Document (PDF/JPG/PNG)", type=['png', 'jpg', 'jpeg', 'pdf'])

# Clear ChromaDB directory
CHROMA_DIR = "./chroma_db"
if os.path.exists(CHROMA_DIR):
    shutil.rmtree(CHROMA_DIR)

if uploaded_file:
    st.success(f"✅ Uploaded: {uploaded_file.name}")
    
    # Now load all the heavy stuff only when needed!
    try:
        # Step 1: Load OCR model
        with st.status("🔍 Loading EasyOCR model (first time may take a few minutes)...", expanded=True) as status:
            try:
                import easyocr
                @st.cache_resource
                def load_ocr():
                    return easyocr.Reader(['bn', 'en'], gpu=False, download_enabled=True)
                reader = load_ocr()
                status.update(label="✅ EasyOCR model loaded!", state="complete", expanded=False)
            except Exception as e:
                status.update(label=f"❌ Error loading OCR: {str(e)}", state="error", expanded=True)
                st.stop()
        
        # Step 2: Load embedding model
        with st.status("📊 Loading embedding model (first time may take a few minutes)...", expanded=True) as status:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                @st.cache_resource
                def load_embeddings():
                    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                embeddings = load_embeddings()
                status.update(label="✅ Embedding model loaded!", state="complete", expanded=False)
            except Exception as e:
                status.update(label=f"❌ Error loading embeddings: {str(e)}", state="error", expanded=True)
                st.stop()
        
        # Step 3: Display the document
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("📄 Uploaded Document")
            if uploaded_file.type == "application/pdf":
                with st.spinner("Loading PDF..."):
                    import pdf2image
                    images = pdf2image.convert_from_bytes(uploaded_file.read())
                    st.image(images[0], caption="First Page of PDF", use_column_width=True)
                    uploaded_file.seek(0)
            else:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Step 4: Extract text
        with col2:
            with st.status("� Extracting text using OCR...", expanded=True) as status:
                try:
                    extracted_text = ""
                    if uploaded_file.type == "application/pdf":
                        import pdf2image
                        images = pdf2image.convert_from_bytes(uploaded_file.read())
                        for i, img in enumerate(images):
                            status.write(f"Processing page {i+1}/{len(images)}")
                            image_np = np.array(img)
                            results = reader.readtext(image_np, detail=0)
                            extracted_text += " ".join(results) + "\n\n"
                    else:
                        image = Image.open(uploaded_file)
                        image_np = np.array(image)
                        results = reader.readtext(image_np, detail=0)
                        extracted_text = " ".join(results)
                    
                    status.update(label="✅ Text extraction complete!", state="complete", expanded=False)
                    st.subheader("📝 Extracted Text")
                    st.text_area("", extracted_text, height=300)
                except Exception as e:
                    status.update(label=f"❌ Error extracting text: {str(e)}", state="error", expanded=True)
                    st.stop()
            
            # Step 5: Chunk and index
            with st.status("🧩 Chunking and indexing document...", expanded=True) as status:
                try:
                    from langchain_text_splitters import RecursiveCharacterTextSplitter
                    from langchain_community.vectorstores import Chroma
                    from rank_bm25 import BM25Okapi
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = text_splitter.split_text(extracted_text)
                    status.write(f"Created {len(chunks)} chunks")
                    
                    metadatas = [{"date": str(doc_date), "type": doc_type, "lang": doc_lang} for _ in chunks]
                    
                    # Initialize ChromaDB (without persist which is deprecated)
                    vectorstore = Chroma.from_texts(
                        chunks, 
                        embeddings, 
                        metadatas=metadatas, 
                        persist_directory=CHROMA_DIR
                    )
                    
                    # Initialize BM25 for keyword search
                    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
                    bm25 = BM25Okapi(tokenized_chunks)
                    status.write("BM25 index created")
                    
                    status.update(label="✅ Document indexed successfully!", state="complete", expanded=False)
                    st.info("Hybrid search (BM25 + vector) is now active!")
                except Exception as e:
                    status.update(label=f"❌ Error indexing: {str(e)}", state="error", expanded=True)
                    st.stop()
        
        # Step 6: RAG query
        st.markdown("---")
        query = st.text_input("💬 Ask a question about the document:")
        
        if query:
            with st.status("🤖 Generating answer with hybrid search...", expanded=True) as status:
                try:
                    from langchain_community.llms import Ollama
                    from langchain_core.prompts import PromptTemplate
                    from langchain_core.documents import Document
                    
                    llm = Ollama(model="llama3")
                    
                    # Step 1: BM25 Keyword Search
                    tokenized_query = query.lower().split()
                    bm25_scores = bm25.get_scores(tokenized_query)
                    
                    # Step 2: Vector Search with filters
                    filtered_chroma_results = vectorstore.similarity_search_with_score(
                        query,
                        k=10,
                        filter={
                            'date': str(doc_date),
                            'type': doc_type,
                            'lang': doc_lang
                        }
                    )
                    
                    # Step 3: Get chunk indices for BM25 that match metadata
                    bm25_chunk_indices = []
                    for i, meta in enumerate(metadatas):
                        if (meta['date'] == str(doc_date) and 
                            meta['type'] == doc_type and 
                            meta['lang'] == doc_lang):
                            bm25_chunk_indices.append(i)
                    
                    # Step 4: Combine scores using Reciprocal Rank Fusion (RRF)
                    rrf_scores = defaultdict(float)
                    k = 60  # RRF constant
                    
                    # Process ChromaDB results (sorted by similarity)
                    for rank, (doc, score) in enumerate(filtered_chroma_results):
                        chunk_idx = chunks.index(doc.page_content)
                        rrf_scores[chunk_idx] += 1 / (k + rank + 1)
                    
                    # Process BM25 results (filter and sort top 10)
                    filtered_bm25 = [(i, bm25_scores[i]) for i in bm25_chunk_indices]
                    filtered_bm25.sort(key=lambda x: x[1], reverse=True)
                    for rank, (i, score) in enumerate(filtered_bm25[:10]):
                        rrf_scores[i] += 1 / (k + rank + 1)
                    
                    # Step 5: Get top chunks based on RRF
                    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
                    top_chunk_indices = [i for i, _ in sorted_rrf[:5]]
                    
                    # Create documents for the chain
                    final_docs = [
                        Document(page_content=chunks[i], metadata=metadatas[i]) 
                        for i in top_chunk_indices
                    ]
                    
                    # Show source info
                    with st.expander("🔍 Hybrid Search Details"):
                        st.write("Top chunks from BM25 + Vector combination:")
                        for i, idx in enumerate(top_chunk_indices):
                            st.markdown(f"**Chunk {i+1}** (RRF Score: {rrf_scores[idx]:.4f})")
                            st.write(chunks[idx][:200] + "...")
                    
                    # Create prompt and generate answer
                    prompt = PromptTemplate.from_template(
                        """Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}

Answer:"""
                    )
                    
                    # Manually run document chain since we have final_docs
                    context = "\n\n".join([doc.page_content for doc in final_docs])
                    response = llm.invoke(prompt.format(context=context, input=query))
                    
                    status.update(label="✅ Answer generated!", state="complete", expanded=False)
                    st.subheader("✅ Answer")
                    st.markdown(response)
                    
                    with st.expander("📚 Source Chunks"):
                        for i, doc in enumerate(final_docs):
                            st.markdown(f"**Chunk {i+1}**")
                            st.write(doc.page_content)
                            st.markdown("---")
                except Exception as e:
                    status.update(label=f"❌ Error with RAG: {str(e)}", state="error", expanded=True)
                    st.info("Make sure Ollama is running and has llama3: `ollama run llama3`")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
