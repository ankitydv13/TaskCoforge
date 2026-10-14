from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import (
    HuggingFaceEndpoint , 
    ChatHuggingFace , 
    HuggingFaceEmbeddings
    )
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from Services.retrival_service import retrival
from Services.reranking_service import reranking_service
from dotenv import load_dotenv
import os

from database import model
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select


load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME")

class LLMService:

    NOT_FOUND_RESPONSE = (
    "I couldn't find this in the available policy documents. "
    "Please check with HR for confirmation."
    )


    def __init__(self, provider: str):
        self.llm = LLMFactory.create(provider)

    async def query_long_term_memory(self, session_id: str, db: AsyncSession):

        if not session_id:
            raise ValueError("session_id is unavailable")

        result_obj = await db.execute(select(model.ChatMessage)
                .where(model.ChatMessage.session_id == session_id)
                .order_by(model.ChatMessage.id))

        all_result = result_obj.scalars().all()

        all_memory = "\n".join(
            [
                f"{chat.role}:{chat.message}"
                for chat in all_result
            ]
        )

        return all_memory



    async def chain(self):
        _answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are an HR policy assistant for Coforge employees. Answer using "
        "ONLY the policy content below. Do not guess, estimate, or use outside "
        "knowledge — every number or fact in your answer must come directly "
        "from the context. Give a concise, direct answer when a source passage contains the answer "
        "Context:\n{context}\n"
        "Retrieved memory:\n{retrieved_memory}"),
        ("human", "{query}"),
        ])    
        parser = StrOutputParser()
        _answer_chain = _answer_prompt | self.llm | parser
        return _answer_chain

    
    async def query(self, query: str, session_id: str, db: AsyncSession):

        long_term_memory = await self.query_long_term_memory(session_id, db)

        _answer_chain = await self.chain()

        print(f"Retriving the chunk before reranking")
        retrive_chunk = retrival.chroma_query(query,str(COLLECTION_NAME))

        documents =  [y for x in retrive_chunk["documents"] for y in x]
        reranked_chunk , rerank_score = reranking_service.cross_encoder_reranking(query,documents)
        
        ans = _answer_chain.invoke({
            "context": reranked_chunk[:5],
            "query": query,
            "retrieved_memory": long_term_memory or "No retrieved memory found for this session",
        })
        return {
            "answer": ans,
            "reranked_chunk": reranked_chunk[:5],
            "rerank_score" : rerank_score[:5]
        }
        
    

class LLMFactory:
    @staticmethod
    def create(provider:str=''):
        if provider == '' or provider == 'huggingface':
            print("Default provider huggingface")
            llm = HuggingFaceEndpoint(
                repo_id='meta-llama/Llama-3.1-8B-Instruct',
                task='text-generation',
                max_new_tokens=350
                )
            
            return ChatHuggingFace(llm=llm)
        
        elif provider == "openai":
            return ChatOpenAI(model = "gpt-5.4-mini")

        elif provider == "openaiembeddings":
            return OpenAIEmbeddings()

        elif provider == "huggingfaceembeddings":
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        raise ValueError("Unsupported provider")
