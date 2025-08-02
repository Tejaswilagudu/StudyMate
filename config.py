class Config:
    # Chunking parameters
    CHUNK_SIZE = 200
    CHUNK_OVERLAP = 50
    
    # Empty placeholders - use environment variables instead
    WATSONX_API_KEY = ""
    HF_API_KEY = ""
    HF_TOKEN=""
    HF_EMAIL="abhinaireddy2244@gmail.com"
    HF_PASSWORD="Abhinai.7902"
    email="abhinaireddy2244@gmail.com"
    password="Abhinai.7902"

    
    # Feature toggles
    USE_SECONDARY_LLM = True
    TRANSLATE_QUERY = False
    
    # Model parameters
    GRANITE_PARAMS = {
        "MAX_NEW_TOKENS": 512,
        "TEMPERATURE": 0.7,
        "TOP_P": 0.9
    }
    
    # Tesseract path
    TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"