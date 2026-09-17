
from pydantic import BaseModel , Field
from typing import Dict , Any

class ChunkResponse(BaseModel):
    chunk_strategy : str 
    no_of_chunk : int 
    page_content : list[str]
    answer : str 


