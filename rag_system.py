import streamlit as st
import easyocr
import numpy as np
import os
from PIL import Image
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import shutil

# OCR Engine Load (Bangla + English)
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['bn', 'en'])

reader = load_ocr()

st.set_page_config(page_title="Local Multilingual RAG", layout="wide")
st.title("📄 Local OCR & Dynamic RAG System")

# Sidebar: Manual Metadata Filters
st.sidebar.header("Manual Metadata Filters")
doc_date = st.sidebar.date_input("Document Date")
doc_type = st.sidebar.selectbox("Document Type", ["Official", "Personal"])
doc_lang = st.sidebar.radio("Document Language", ["Bangla", "English", "Mixed"])

uploaded_file = st.file_uploader("Upload Scanned Image (JPG/PNG)", type=['png', 'jpg', 'jpeg'])

# Clear ChromaDB directory to avoid conflicts
CHROMA_DIR = "./chroma_db"
if os.path.exists(CHROMA_DIR):
    shutil.rmtree(CHROMA_DIR)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Document", width=400)
    
    with st.spinner("Step 1: Extracting text locally..."):
        image_np = np.array(image)
        results = reader.readtext(image_np, detail=0)
        extracted_text = " ".join(results)
        st.subheader("Extracted Text Log:")
        st.text_area("", extracted_text, height=200)

    # Chunking and Embeddings (Multilingual)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(extracted_text)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    metadatas = [{"date": str(doc_date), "type": doc_type, "lang": doc_lang} for _ in chunks]
    vectorstore = Chroma.from_texts(chunks, embeddings, metadatas=metadatas, persist_directory=CHROMA_DIR)
    vectorstore.persist()
    st.success("Step 2: Indexed in Local Vector DB with Metadata!")

    # Dynamic RAG Search with Full Metadata Filtering
    query = st.text_input("Ask a question about the document:")
    
    if query:
        with st.spinner("Step 3: Ollama (Llama 3) is thinking..."):
            llm = Ollama(model="llama3")
            
            # Apply all metadata filters
            retriever = vectorstore.as_retriever(
                search_kwargs={
                    'filter': {
                        'date': str(doc_date),
                        'type': doc_type,
                        'lang': doc_lang
                    }
                }
            )
            
            # Create prompt and chains using updated LangChain API
            prompt = PromptTemplate.from_template(
                """Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}"""
            )
            
            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            response = retrieval_chain.invoke({"input": query})
            
            st.subheader("Answer:")
            st.write(response["answer"])
