import uuid

import faiss
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

import embedding

VECTOR_DB_PATH = "vector_db"

def create_vector_store(documents):

    chunks_doc = [document.page_content
             for document in documents]

    vectors = embedding.embedding.embed_documents(chunks_doc)

    dimension = len(vectors[0])

    print(f"dimension of the vector {dimension}")

    vector_np = np.ndarray(
        vectors,
        dtype= np.float32
    )

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

    vector_store.save_local(
        VECTOR_DB_PATH
    )

    print(
        "Vector store saved successfully."
    )

def load_vector_store():

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





