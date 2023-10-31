import os

import openai
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

print(sys.version)

# 실제 사용시엔 debug false로
app = FastAPI(debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    message: str
    temperature: float = 1

SYSTEM_MSG = "You are a helpful travel assistant, Your name is Jini, 27 years old."

@app.post("/chat")
def chat(req: ChatRequest):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system", "content": SYSTEM_MSG
        },
        {
            "role": "user", "content": req.message
        }
    ],
    # 랜덤성 : 0 ~ 1 사이의 값으로 0에 가까울수록 고정된 답변을 반환하게 됨
    temperature=req.temperature
    )

    return {"message": response.choices[0].message.content}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)