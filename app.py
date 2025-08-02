import streamlit as st
import pandas as pd
import os
from config import Config
from indexer import process_pdfs
from querier import generate_answer, generate_summary, translate_text
from models import load_models, init_granite_model, init_hugchat

# Streamlit page configuration
st.set_page_config(page_title="StudyMate: AI-Powered Study Assistant", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to StudyMate! Upload your PDFs and ask questions to get started."}]
if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = {}
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chunk_metadata" not in st.session_state:
    st.session_state.chunk_metadata = []

# Load models
try:
    embedder, summarizer, translator = load_models()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# Sidebar for configuration
with st.sidebar:
    st.title("StudyMate Configuration")
    st.markdown("### Upload PDFs")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    st.markdown("### API Keys")
    watsonx_api_key = st.text_input("IBM Watsonx API Key", type="password", value=os.getenv("WATSONX_API_KEY", Config.WATSONX_API_KEY))
    hf_api_key = st.text_input("Hugging Face API key (Optional)", type="password", value=os.getenv("HF_API_KEY", Config.HF_API_KEY))
    st.markdown("### Settings")
    chunk_size = st.slider("Chunk Size (words)", 100, 500, Config.CHUNK_SIZE)
    chunk_overlap = st.slider("Chunk Overlap (words)", 0, 100, Config.CHUNK_OVERLAP)
    use_secondary_llm = st.checkbox("Use Secondary LLM (HugChat)", value=Config.USE_SECONDARY_LLM)
    translate_query = st.checkbox("Translate Query to English", value=Config.TRANSLATE_QUERY)

# Initialize LLMs
granite_model = init_granite_model(watsonx_api_key)
hugchat_model = init_hugchat(hf_api_key) if use_secondary_llm else None

# Main UI
st.title("StudyMate: AI-Powered Study Assistant")
st.markdown("Upload academic PDFs, ask questions, and get contextual answers powered by IBM Granite and RAG.")

# Process uploaded PDFs
if uploaded_files:
    with st.spinner("Processing PDFs..."):
        try:
            process_pdfs(uploaded_files, chunk_size, chunk_overlap, embedder)
            st.success("PDFs processed successfully!")
        except Exception as e:
            st.error(f"Failed to process PDFs: {e}")

# Display summaries
if st.session_state.pdf_data:
    st.subheader("PDF Summaries")
    for pdf_name, text_data in st.session_state.pdf_data.items():
        full_text = " ".join(page["text"] for page in text_data)
        with st.expander(f"Summary for {pdf_name}"):
            st.write(generate_summary(full_text, summarizer))

# Chat interface
st.subheader("Ask a Question")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "source" in msg:
            st.markdown(f"*Source: {msg['source']}*")

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Translate query if enabled
    if translate_query:
        try:
            prompt = translate_text(prompt, translator)
            st.info(f"Translated query: {prompt}")
        except Exception as e:
            st.warning(f"Translation failed: {e}")
    
    # Retrieve and generate answer
    if st.session_state.faiss_index and (granite_model or hugchat_model):
        with st.spinner("Generating answer..."):
            try:
                answer, llm_used, source_info = generate_answer(
                    prompt, st.session_state.chunks, st.session_state.chunk_metadata,
                    granite_model, hugchat_model, embedder, st.session_state.faiss_index
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "source": source_info
                })
                with st.chat_message("assistant"):
                    st.write(answer)
                    st.markdown(f"*Source: {source_info}*")
                    st.write(f"*Powered by: {llm_used}*")
            except Exception as e:
                st.error(f"Failed to generate answer: {e}")
    else:
        st.error("Please upload PDFs and provide at least one LLM API key.")

# Download Q&A log
if st.session_state.messages:
    log_data = [{"Role": msg["role"], "Content": msg["content"], "Source": msg.get("source", "")} for msg in st.session_state.messages]
    log_df = pd.DataFrame(log_data)
    csv = log_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Q&A Log",
        data=csv,
        file_name="studymate_qa_log.csv",
        mime="text/csv"
    )

# User feedback
st.subheader("Rate Your Experience")
rating = st.slider("How helpful was the response? (1 = Poor, 5 = Excellent)", 1, 5, 3)
if st.button("Submit Feedback"):
    st.success(f"Thank you for your {rating}-star rating!")

if __name__ == "__main__":
    st.write("StudyMate is ready to assist you!")