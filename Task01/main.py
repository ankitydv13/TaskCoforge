from pathlib import Path
import shutil

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
from schema import ChunkResponse , ChunkConfig

from faiss_vector_store import (
    create_vector_store,
    load_vector_store,
    save_vector_store,
    faiss_with_cosine,
    to_numpy_arr,
    display_vector_db
)

from embedding import embedding
import faiss

import json

load_dotenv()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI()


@app.post("/api/upload")
def upload_file(
    technique_name: ChunkingStrategy = 
        ChunkingStrategy.CharacterChunk,
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
    

    #TODO : Choose one 
    # vector_store = create_vector_store(result)
    vector_store = faiss_with_cosine(result)
    display_vector_db(vector_store,20)

    # save_vector_store(vector_store)

    query_vector = embedding.embed_query(query)
    query_np = to_numpy_arr([query_vector])
    faiss.normalize_L2(query_np)

    query_result = vector_store.search(query , "similarity")

    # query_result = vector_store.similarity_search(query , k=2)

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