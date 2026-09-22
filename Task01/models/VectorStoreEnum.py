from enum import Enum

class VectorStoreEnum(str,Enum):
    faiss = "faiss"
    faiss_cosin = "faiss with cosine similarity"
    chromadb = "chromadb"