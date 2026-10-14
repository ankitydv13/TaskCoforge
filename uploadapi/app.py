from fastapi import FastAPI
from routers.pdfroutes.pdf_router import router as pdf_router
from routers.chatbotroute.chatbot_routes import router as chatbot_router
from routers.EvaluationRoutes.evaluation_routes import router as eval_router

from contextlib import asynccontextmanager
from Services.llm_services import LLMService
from database import model
from database.database import engine

import os
import shutil
from Configs.config import settings

@asynccontextmanager
async def lifespan(app_:FastAPI):
    print("Connection to database")
    async with engine.begin() as conn:
        await conn.run_sync(model.Base.metadata.create_all)

    print("Creating Object Of LLM Services")
    app_.state.llm_service = LLMService("openai")
    
    try:
        yield
    finally:
        print("Shutting Down...")
        await engine.dispose()

app = FastAPI(lifespan=lifespan)

app.include_router(pdf_router)
app.include_router(chatbot_router)
app.include_router(eval_router)

UPLOAD_DIR = settings.upload_dir

@app.on_event("shutdown")
def clear_upload_dir():
    """Clear the upload directory when FastAPI shuts down."""
    if os.path.exists(UPLOAD_DIR):
        # Remove all files inside the directory
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
        print("Upload directory cleared.")
