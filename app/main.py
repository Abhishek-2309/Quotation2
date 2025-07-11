from fastapi import FastAPI
from app.routes import router

app = FastAPI(title="Document Field Extractor")

app.include_router(router, prefix="/api")

