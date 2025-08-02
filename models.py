from sentence_transformers import SentenceTransformer
from transformers import pipeline
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from hugchat import hugchat
from hugchat.login import Login
from config import Config

def load_models():
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")
        return embedder, summarizer, translator
    except Exception as e:
        raise Exception(f"Failed to load models: {e}")

def init_granite_model(api_key):
    if not api_key:
        return None
    try:
        model = Model(
            model_id="ibm/granite-13b-chat",
            credentials={"api_key": api_key, "url": "https://us-south.ml.cloud.ibm.com"},
            params=Config.GRANITE_PARAMS
        )
        return model
    except Exception as e:
        print(f"Failed to initialize IBM Granite: {e}")
        return None

def init_hugchat(hf_token=None):
    if not hf_token:
        return None
    try:
        # For hugchat >= 0.4.0
        sign = Login()
        cookies = sign.loginWithHuggingFace(hf_token)
        chatbot = hugchat.ChatBot(cookies=cookies)
        return chatbot
    except Exception as e:
        print(f"Failed to initialize HugChat: {e}")
        return None