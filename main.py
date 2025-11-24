# src/app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.models.predictor import ModelPredictor
import os
from starlette.responses import FileResponse, HTMLResponse

app = FastAPI(title="Fake News & Spam Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

predictor = ModelPredictor()

class TextIn(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
def root():
    index_path = os.path.abspath(os.path.join(os.getcwd(), 'web', 'index.html'))
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Fake News & Spam Classifier API - visit /docs for Swagger UI"}

@app.get("/models")
def list_models():
    return {"models": predictor.available_models()}

@app.post("/predict/{task}")
def predict(task: str, item: TextIn):
    if task not in ['spam', 'fakenews']:
        raise HTTPException(status_code=400, detail="task must be 'spam' or 'fakenews'")
    return predictor.predict(item.text, task=task)

