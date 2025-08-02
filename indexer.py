import fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st
from PIL import Image
import pytesseract
from config import Config

def extract_text_from_pdf(pdf_file):
    text_data = []
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if not text.strip():
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.rgb)
                text = pytesseract.image_to_string(img)
            text_data.append({"page": page_num + 1, "text": text})
        doc.close()
    except Exception as e:
        st.error(f"Failed to extract text from PDF: {e}")
    return text_data

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def process_pdfs(uploaded_files, chunk_size, overlap, embedder):
    st.session_state.pdf_data = {}
    st.session_state.chunks = []
    st.session_state.chunk_metadata = []
    
    # Set Tesseract path if specified
    if Config.TESSERACT_PATH:
        pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_PATH
    
    for pdf_file in uploaded_files:
        text_data = extract_text_from_pdf(pdf_file)
        if text_data:
            st.session_state.pdf_data[pdf_file.name] = text_data
            for page_data in text_data:
                chunks = chunk_text(page_data["text"], chunk_size, overlap)
                for chunk in chunks:
                    st.session_state.chunks.append(chunk)
                    st.session_state.chunk_metadata.append({
                        "pdf_name": pdf_file.name,
                        "page": page_data["page"]
                    })
    
    print(f"Extracted chunks: {len(st.session_state.chunks)}") 
    
    if st.session_state.chunks:
        try:
            embeddings = embedder.encode(st.session_state.chunks, show_progress_bar=True)
            dimension = embeddings.shape[1]
            st.session_state.faiss_index = faiss.IndexFlatL2(dimension)
            st.session_state.faiss_index.add(embeddings.astype(np.float32))
        except Exception as e:
            st.error(f"Failed to create FAISS index: {e}")