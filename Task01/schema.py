
from pydantic import BaseModel , Field , model_validator , field_validator 
from langchain_core.documents import Document
from typing import Dict, Any

class ChunkResponse(BaseModel):
    chunk_strategy : str 
    no_of_chunk : int 
    page_content : list[str]
    answer : str

class ChunkConfig(BaseModel):
    metadata: Dict[str, Any]
    chunk_size: int
    chunk_overlap: int
    separators: str

    @model_validator(mode="after")
    def validate_chunk_config(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be less than chunk_size"
            )
        return self

    @field_validator("separators")
    @classmethod
    def validate_separator(cls, value):
        valid_separators = {"\n\n", "\n", " ", ""}
        if value not in valid_separators:
            raise ValueError(
                "separator must contain only '\\n\\n', '\\n', ' ', or ''"
            )

        return value
    
class UpdateChunkRequest(BaseModel):
    doc_id: str
    new_text: str

class UpdatePolicyRequest(BaseModel):
    search_text: str     
    new_text: str    
