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
)

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from semantic_chunker_langchain.chunker import SemanticChunker

from models.ChunkingStrategyEnum import ChunkingStrategy
from schema import ChunkResponse

from faiss_vector_store import (
    create_vector_store,
    load_vector_store,
    save_vector_store
)

load_dotenv()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI()


@app.post("/api/upload", response_model=ChunkResponse , status_code=status.HTTP_200_OK)
def upload_file(
    file: UploadFile = File(...),
    technique_name: ChunkingStrategy = Query(
        ChunkingStrategy.CharacterChunk
    ),
    chunk_size: int = 100,
    chunk_overlap: int = 20,
    seprator : str = "",
    query : str = ""
):
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found",
        )
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST ,
            detail="Chunk Overlap must be less than and not equal to Chunk Size"
        )

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
# TODO : Update the package according to file type
    loader = UnstructuredFileLoader(
        str(file_path),
        mode="elements",
        strategy="fast",
    )

    docs = loader.load()

    if technique_name == ChunkingStrategy.CharacterChunk:
        result = character_chunking(
            docs,
            chunk_size,
            chunk_overlap,
            seprator
        )

    elif technique_name == ChunkingStrategy.RecurChunk:
        result = recursive_chunking(
            docs,
            chunk_size,
            chunk_overlap,
            seprator
        )

    elif technique_name == ChunkingStrategy.Semantic:
        result = semantic_chunking(docs)

    elif technique_name == ChunkingStrategy.Token:
        result = token_chunking(
            docs,
            chunk_size,
            chunk_overlap,
        )
    print(type(result))

    vector_store = create_vector_store(result)

    save_vector_store(vector_store)

    query_result = vector_store.similarity_search(query , k=2)

    page_content = [doc.page_content for doc in result]

    print()
    print(page_content)

    chunk_response = ChunkResponse(
        chunk_strategy=technique_name,
        no_of_chunk=len(page_content),
        page_content=page_content,
        answer= query_result
    )

    return chunk_response


def recursive_chunking(
    docs: list,
    size: int,
    overlap: int,
    seprator : str
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        seprator = seprator
    )

    return splitter.split_documents(docs)


def character_chunking(
    docs: list,
    size: int,
    overlap: int,
    seprator : str
):
    splitter = CharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separator=seprator,
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