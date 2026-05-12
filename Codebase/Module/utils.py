from sentence_transformers import SentenceTransformer
from .config import MODEL_NAME

class ModelLoader:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_model(cls):
        if cls._model is None:
            print(f"  [System] Loading Semantic Engine ({MODEL_NAME})...")
            cls._model = SentenceTransformer(MODEL_NAME)
        return cls._model
