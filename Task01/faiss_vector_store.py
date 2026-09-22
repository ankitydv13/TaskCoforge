import uuid

import math

import faiss
from networkx import display
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

import pandas as pd

import embedding

VECTOR_DB_PATH = "vector_db"

def create_vector_store(documents):

    chunks_doc = [document.page_content
             for document in documents]

    vectors = embedding.embedding.embed_documents(chunks_doc)

    dimension = len(vectors[0])

    print(f"dimension of the vector {dimension}")

    vector_np = to_numpy_arr(vectors)

    print(f"Vector shape {vector_np.shape}")

    #TODO: choose the no of the cluster accordling 
    nlist = 2

    quantize = faiss.IndexFlatL2(dimension)

    index = faiss.IndexIVFFlat(
        quantize,
        dimension,
        nlist,
        faiss.METRIC_L2
    )

    index.train(vector_np)

    index.add(vector_np)

    print(f"Vector add in store {index.ntotal}")

    index.nprobe =1 

    docstore = InMemoryDocstore()

    index_to_docstore_id = {}

    for faiss_id , document in enumerate(documents):

        doc_id = str(uuid.uuid4())

        docstore.add({
            doc_id : document
        })

        index_to_docstore_id[faiss_id] = doc_id

    vector_store = FAISS(
        embedding_function = embedding.embedding,
        index = index,
        docstore = docstore,
        index_to_docstore_id = index_to_docstore_id
    )

    return vector_store

def save_vector_store(vector_store):
    print("Saving vector store to local path...",VECTOR_DB_PATH)
    vector_store.save_local(
        VECTOR_DB_PATH
    )

    print(
        "Vector store saved successfully."
    )

def load_vector_store():
    print("loading vector store from local path...",VECTOR_DB_PATH)
    vector_store = FAISS.load_local(
        VECTOR_DB_PATH,
        embedding.embedding,
        allow_dangerous_deserialization=True
    )

    # Configure IVF search
    vector_store.index.nprobe = 1

    print(
        "Vector store loaded successfully."
    )

    return vector_store



def faiss_with_cosine(documents):
    chunks_doc = [document.page_content
                 for document in documents]

   
    
    vectors = embedding.embedding.embed_documents(chunks_doc)

    vector_np = to_numpy_arr(vectors)

    faiss.normalize_L2(vector_np)

    dimension = vector_np.shape[1]

    print(dimension) #384

    #TODO: nlist is sq root of no of vector 
    nlist = 2

    
    index = faiss.IndexFlatIP(dimension)
    index.add(vector_np)

    print(f"total index vector {index.ntotal} ")

    

    docstore = InMemoryDocstore()

    index_to_docstore_id = {}

    for faiss_id , document in enumerate(documents):

        doc_id = str(uuid.uuid4())

        docstore.add({
            doc_id : document
        })

        index_to_docstore_id[faiss_id] = doc_id

    vector_store = FAISS(
        embedding_function = embedding.embedding,
        index = index,
        docstore = docstore,
        index_to_docstore_id = index_to_docstore_id,
        normalize_L2 = True
    )

    

    return vector_store

def to_numpy_arr(vectors):
    return np.array(
        vectors,
        dtype = np.float32
    )

def display_vector_db(vector_store, n):
    v_dict = vector_store.docstore._dict   # fix: actual doc_id -> Document mapping

    data_row = []
    for doc_id, document in v_dict.items():
        source = document.metadata.get("source", "").split("/")[-1]
        page_no = document.metadata.get("page", "")
        title = document.metadata.get("title", "")
        chunk = document.page_content

        data_row.append({
            "faiss_id": doc_id,
            "source": source,
            "page_no": page_no,
            "title": title,
            "chunk": chunk
        })

    v_df = pd.DataFrame(data_row)
    print(v_df.head(n))
    print("length of the dataframe:", len(v_df))

    


if __name__ == "__main__":
    from langchain_core.documents import Document

    documents = [
        Document(
            page_content="LangChain makes building LLM-powered apps easier.",
            metadata={"source": "blog_post", "author": "LangChain Team", "date": "2025-01-10"}
        ),
        Document(
            page_content="Python is an interpreted, high-level programming language.",
            metadata={"source": "wikipedia", "topic": "programming", "language": "en"}
        ),
        Document(
            page_content="Acme Corp quarterly revenue grew 12% in Q2 2025.",
            metadata={"source": "financial_report.pdf", "page": 4, "year": 2025}
        ),
        Document(
            page_content="The Eiffel Tower is located in Paris, France.",
            metadata={"source": "travel_guide", "page": 12, "tags": ["landmark", "France"]}
        ),
        Document(
            page_content="SQLAlchemy is a Python ORM library for database interactions.",
            metadata={"source": "developer_docs", "section": "ORM", "version": "2.0"}
        )
    ]
    vector_store = faiss_with_cosine(documents)
    query_results = vector_store.search("where is Eiffel Tower", "similarity")
    print(type(query_results))
    print(query_results)
    # display_vector_db(vector_store, 5)

