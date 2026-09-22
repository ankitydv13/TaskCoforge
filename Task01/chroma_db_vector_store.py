from pathlib import Path
from typing import List
import uuid
 
from langchain_chroma import Chroma
from langchain_core.documents import Document
 
from embedding import embedding
 
CHROMA_DIR = Path("chroma_db")
CHROMA_DIR.mkdir(exist_ok=True)
 
COLLECTION_NAME = "pdf_chunks"
 
 
def get_chroma_store() -> Chroma:
    
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=str(CHROMA_DIR),
    )
 
 
def add_documents_to_chroma(documents: List[Document], file_id: str) -> List[str]:
    
    vector_store = get_chroma_store()
 
    ids = [str(uuid.uuid4()) for _ in documents]
 
    # tag every chunk with file_id so we can filter/delete/update by source later
    for doc in documents:
        doc.metadata["file_id"] = file_id
 
    vector_store.add_documents(documents=documents, ids=ids)
 
    return ids
 
 
def update_document_by_id(doc_id: str, new_text: str, new_metadata: dict = None) -> None:
    
    vector_store = get_chroma_store()
 
    existing = vector_store.get(ids=[doc_id])
    if not existing["ids"]:
        raise ValueError(f"No document found with id={doc_id}")
 
    old_metadata = existing["metadatas"][0]
    merged_metadata = {**old_metadata, **(new_metadata or {})}
 
    updated_doc = Document(page_content=new_text, metadata=merged_metadata)
 
    vector_store.update_document(document_id=doc_id, document=updated_doc)
 
 
def find_chunks_by_metadata(key: str, value) -> dict:
    
    vector_store = get_chroma_store()
    return vector_store.get(where={key: value})
 
 
def delete_chunks_by_metadata(key: str, value) -> int:
    
    vector_store = get_chroma_store()
    matches = vector_store.get(where={key: value})
    ids_to_delete = matches["ids"]
 
    if ids_to_delete:
        vector_store.delete(ids=ids_to_delete)
 
    return len(ids_to_delete)