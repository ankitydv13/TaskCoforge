from enum import Enum

class ChunkingStrategy(str,Enum):
    CharacterChunk = "Character Chunking"
    RecurChunk = "Recursive Chunking"
    Semantic = "Semantic Chunking"
    Token = "Token Chunking"