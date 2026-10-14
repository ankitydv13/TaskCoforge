from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from Schemas.schemas import ChatSessionResponse, LLMResponse

import os 
from dotenv import load_dotenv

from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated
from database import model
from sqlalchemy import select
from database.database import get_db

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME")

router = APIRouter(tags=["chatbot"])


@router.post("/chat-sessions", response_model=ChatSessionResponse, status_code=201)
async def create_chat_session(
    db: Annotated[AsyncSession, Depends(get_db)],
):
    chat_session = model.ChatSession(session_id=str(uuid4()))
    db.add(chat_session)
    await db.commit()
    await db.refresh(chat_session)
    return ChatSessionResponse(session_id=chat_session.session_id)


@router.post("/chatbot",response_model=LLMResponse)
async def askchatbot(
    request: Request,
    query: str,
    session_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    print("*"*20 + "askchatbot" + "*"*20)

    result_obj = await db.execute(select(model.ChatSession)
                        .where(model.ChatSession.session_id == session_id))
    if result_obj.scalars().first() is None:
        raise HTTPException(status_code=404, detail="Chat session not found")

    llm_services = request.app.state.llm_service

    result = await llm_services.query(query, session_id, db)
    db.add_all([
        model.ChatMessage(session_id=session_id, role="user", message=query),
        model.ChatMessage(
            session_id=session_id,
            role="assistant",
            message=result["answer"],
        ),
    ])
    await db.commit()

    return LLMResponse(
        query=query,
        answer=result["answer"],
        chunks=result["reranked_chunk"],
        scores=result["rerank_score"]
    )