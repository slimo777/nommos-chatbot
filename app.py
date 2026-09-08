from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from answer import answer

app = FastAPI()


class Turn(BaseModel):
    role: str
    text: str


class Question(BaseModel):
    message: str
    history: list[Turn] = []


@app.post("/chat")
def chat(q: Question):
    history = [{"role": t.role, "text": t.text} for t in q.history]
    return {"reply": answer(q.message, history)}


@app.get("/")
def index():
    return FileResponse("index.html")