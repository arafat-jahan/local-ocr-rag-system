import streamlit as st
import numpy as np
import os
from PIL import Image
import shutil

# Set page config FIRST!
st.set_page_config(page_title="Local Multilingual RAG", layout="wide")

# App title and basic UI
st.title("📄 Local OCR & Dynamic RAG System")
st.markdown("---")

st.info("This app will load OCR and embedding models when you upload your first document!")

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
    st.info("Now let's process this document! (Models will be loaded now)")
    
    # Now load all the heavy stuff only when needed!
    try:
        import easyocr
        import pdf2image
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.llms import Ollama
        from langchain_core.prompts import PromptTemplate
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains.retrieval import create_retrieval_chain
        
        # Load OCR
        @st.cache_resource
        def load_ocr():
            return easyocr.Reader(['bn', 'en'], gpu=False)
        
        # Load embeddings
        @st.cache_resource
        def load_embeddings():
            return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        reader = load_ocr()
        embeddings = load_embeddings()
        
        # Display the document
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("📄 Uploaded Document")
            if uploaded_file.type == "application/pdf":
                with st.spinner("Loading PDF..."):
                    images = pdf2image.convert_from_bytes(uploaded_file.read())
                    st.image(images[0], caption="First Page of PDF", use_column_width=True)
                    uploaded_file.seek(0)
            else:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Extract text
        with col2:
            with st.spinner("🔍 Extracting text..."):
                extracted_text = ""
                if uploaded_file.type == "application/pdf":
                    images = pdf2image.convert_from_bytes(uploaded_file.read())
                    for i, img in enumerate(images):
                        st.info(f"Processing page {i+1}...")
                        image_np = np.array(img)
                        results = reader.readtext(image_np, detail=0)
                        extracted_text += " ".join(results) + "\n\n"
                else:
                    image = Image.open(uploaded_file)
                    image_np = np.array(image)
                    results = reader.readtext(image_np, detail=0)
                    extracted_text = " ".join(results)
                
                st.subheader("📝 Extracted Text")
                st.text_area("", extracted_text, height=300)
            
            # Chunk and embed
            with st.spinner("🧩 Indexing into vector DB..."):
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = text_splitter.split_text(extracted_text)
                
                metadatas = [{"date": str(doc_date), "type": doc_type, "lang": doc_lang} for _ in chunks]
                vectorstore = Chroma.from_texts(
                    chunks, 
                    embeddings, 
                    metadatas=metadatas, 
                    persist_directory=CHROMA_DIR
                )
                vectorstore.persist()
                st.success("✅ Document indexed! Now ask questions!")
        
        # RAG query
        st.markdown("---")
        query = st.text_input("💬 Ask a question about the document:")
        
        if query:
            with st.spinner("🤖 Generating answer..."):
                try:
                    llm = Ollama(model="llama3")
                    
                    retriever = vectorstore.as_retriever(
                        search_kwargs={
                            'filter': {
                                'date': str(doc_date),
                                'type': doc_type,
                                'lang': doc_lang
                            }
                        }
                    )
                    
                    prompt = PromptTemplate.from_template(
                        """Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}

Answer:"""
                    )
                    
                    document_chain = create_stuff_documents_chain(llm, prompt)
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)
                    response = retrieval_chain.invoke({"input": query})
                    
                    st.subheader("✅ Answer")
                    st.markdown(response['answer'])
                    
                    with st.expander("📚 Source Chunks"):
                        for i, doc in enumerate(response['context']):
                            st.markdown(f"**Chunk {i+1}**")
                            st.write(doc.page_content)
                            st.markdown("---")
                except Exception as e:
                    st.error(f"Error with RAG: {e}")
                    st.info("Make sure Ollama is running and has llama3: `ollama run llama3`")
    except Exception as e:
        st.error(f"Error processing document: {e}")
