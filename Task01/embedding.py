
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


if not os.getenv("HF_TOKEN"):
    raise ValueError("Api key is not set ")


embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)


if __name__ == "__main__":
    vector = embedding.embed_query("Who am I ?")
    print(vector)