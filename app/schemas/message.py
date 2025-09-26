from pydantic import BaseModel

class Message(BaseModel):
    user_id: str
    text: str


class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    response: str
    backend: str