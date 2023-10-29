import os

import openai
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
openai.api_key = os.getenv("openai_api_key")

# 챗봇 2세대의 경우 고객의 의도(여행 계획, 문의) 등을 파악하여 그에 맞는 작동을 하도록 구현

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

def classify_intent(msg):
    prompt = f"""Your job is to classify intent.

    Choose one of the following intents:
    - travel_plan
    - customer_support
    - reservation

    User: {msg}
    Intent:
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

@app.post("/chat")
def chat(req: ChatRequest):

    intent = classify_intent(req.message)

    if intent == "travel_plan":
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
    
    elif intent == "customer_support":
        return {"message": "Here is customer support number : 1234516"}
    
    elif intent == "resevation":
        return {"message": "Here is reservation number : 1234ae516"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)