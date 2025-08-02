from langdetect import detect
import numpy as np
import streamlit as st

def generate_summary(text, summarizer):
    try:
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        return f"Summary generation failed: {e}"

def translate_text(text, translator):
    try:
        lang = detect(text)
        if lang != "en":
            translated = translator(text)[0]['translation_text']
            return translated
        return text
    except Exception as e:
        return f"Translation failed: {e}"

def generate_answer(query, chunks, metadata, granite_model, hugchat_model, embedder, faiss_index):
    try:
        query_embedding = embedder.encode([query])
        distances, indices = faiss_index.search(query_embedding.astype(np.float32), k=5)
        relevant_chunks = [chunks[i] for i in indices[0]]
        relevant_metadata = [metadata[i] for i in indices[0]]
        
        prompt = f"Answer the following question based on the provided context. If the context is insufficient, say so and provide a general answer if possible.\n\nQuestion: {query}\n\nContext:\n" + "\n".join(relevant_chunks)
        
        if granite_model:
            response = granite_model.generate_text(prompt)
            llm_used = "IBM Granite"
        elif hugchat_model:
            response = hugchat_model.chat(prompt)
            response = str(response)
            llm_used = "HugChat"
        else:
            return "No LLM available. Please provide API keys.", None, ""
        
        source_info = "\n".join([f"{m['pdf_name']}, Page {m['page']}" for m in relevant_metadata])
        return response, llm_used, source_info
    except Exception as e:
        st.error(f"Answer generation failed: {e}")
        return f"Failed to generate answer: {e}", None, ""