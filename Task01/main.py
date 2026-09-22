from pathlib import Path
import shutil
import os


from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
    Form
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker

from models.ChunkingStrategyEnum import ChunkingStrategy 
from models.VectorStoreEnum import VectorStoreEnum
from schema import ChunkResponse , ChunkConfig , UpdateChunkRequest , UpdatePolicyRequest

from faiss_vector_store import (
    create_vector_store,
    load_vector_store,
    save_vector_store,
    faiss_with_cosine,
    to_numpy_arr,
    display_vector_db
)

from chroma_db_vector_store import (
    add_documents_to_chroma,
    update_document_by_id,
    find_chunks_by_metadata,
    delete_chunks_by_metadata,
    get_chroma_store
)

from embedding import embedding
import faiss

import json
import uuid

load_dotenv()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI()


@app.post("/api/upload")
def upload_file(
    technique_name: ChunkingStrategy = 
        ChunkingStrategy.CharacterChunk,
    vector_db : VectorStoreEnum = 
        VectorStoreEnum.faiss,
    query : str = Query(...),
    chunk_config: str = Form(...),
    file: UploadFile = File(...)
):
    chunk_config_obj = ChunkConfig.model_validate(
        json.loads(chunk_config)
    )
    chunk_size = chunk_config_obj.chunk_size
    chunk_overlap = chunk_config_obj.chunk_overlap
    metadata = chunk_config_obj.metadata
    separator = chunk_config_obj.separators
    
    # import pdb; pdb.set_trace()
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found",
        )


    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
# TODO : Update the package according to file type
    loader = PyPDFLoader(
        str(file_path)
    )

    docs = loader.load()
    
    
    

    if technique_name == ChunkingStrategy.CharacterChunk:
        result = character_chunking(
            docs,
            chunk_size,
            chunk_overlap,
            separator
        )

    elif technique_name == ChunkingStrategy.RecurChunk:
        result = recursive_chunking(
            docs,
            chunk_size,
            chunk_overlap,
            separator
        )

    elif technique_name == ChunkingStrategy.Semantic:
        result = semantic_chunking(docs)

    elif technique_name == ChunkingStrategy.Token:
        result = token_chunking(
            docs,
            chunk_size,
            chunk_overlap,
        )
    
    if vector_db == "faiss":
        vector_store = create_vector_store(result)
        save_vector_store(vector_store)
        query_result = vector_store.similarity_search(query , k=2)
    elif vector_db == "faiss with cosine similarity":
        vector_store = faiss_with_cosine(result)
        display_vector_db(vector_store,20)
        query_vector = embedding.embed_query(query)
        query_np = to_numpy_arr([query_vector])
        faiss.normalize_L2(query_np)
        query_result = vector_store.search(query , "similarity")
    elif vector_db == "chromadb":
        file_id = str(uuid.uuid4())
        if os.path.isdir("chroma_db") and os.listdir("chroma_db"):
            chunk_ids = add_documents_to_chroma(result, file_id=file_id)
            print(chunk_ids)
        vector_store = get_chroma_store()
        query_result = vector_store.similarity_search(query,k=2)

    answer = "\\n \\n".join(doc.page_content for doc in query_result)

    page_content = [doc.page_content for doc in result]

    print(answer)

    chunk_response = ChunkResponse(
        chunk_strategy=technique_name,
        no_of_chunk=len(page_content),
        page_content=page_content,
        answer= answer
    )

    return chunk_response

@app.put("/api/documents/chunk")
def update_chunk(payload: UpdateChunkRequest):
    try:
        update_document_by_id(payload.doc_id, payload.new_text)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return {"status": "updated", "doc_id": payload.doc_id}


@app.put("/api/documents/policy-update")
def update_policy(payload: UpdatePolicyRequest):
    vector_store = get_chroma_store()
 
    # semantic search to find the chunk that talks about this policy
    results = vector_store.similarity_search(payload.search_text, k=1)
 
    if not results:
        raise HTTPException(status_code=404, detail="No matching chunk found")
 
    target_doc = results[0]
 
    # find its actual Chroma id (similarity_search doesn't return ids directly,
    # so we match on content — better: store a stable chunk_id in metadata at insert time)
    all_matches = vector_store.get(where_document={"$contains": payload.search_text})
 
    if not all_matches["ids"]:
        raise HTTPException(status_code=404, detail="No matching chunk found")
 
    doc_id = all_matches["ids"][0]
 
    update_document_by_id(doc_id, payload.new_text)
 
    return {"status": "updated", "doc_id": doc_id, "old_text": target_doc.page_content}



def recursive_chunking(
    docs: list,
    size: int,
    overlap: int,
    separator : str
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separator = separator
    )

    return splitter.split_documents(docs)


def character_chunking(
    docs: list,
    size: int,
    overlap: int,
    separator : str
):
    splitter = CharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separator=separator,
    )

    return splitter.split_documents(docs)


def semantic_chunking(docs: list):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=3,
    )

    return splitter.split_documents(docs)


def token_chunking(
    docs: list,
    size: int,
    overlap: int,
):
    splitter = TokenTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
    )

    return splitter.split_documents(docs)

