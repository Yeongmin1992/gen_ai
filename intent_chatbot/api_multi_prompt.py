from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI
from multi_prompt_chains import multi_prompt_chain
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

# fastapi post 요청시 request를 편리하게 담아서 사용하기 위해 BaseModel 사용
class UserRequest(BaseModel):
    user_message: str

@app.post("/qna")
def generate_answer(req: UserRequest) -> Dict[str, str]:
    # multipromptchain은 반드시 input키로 넣어줘야 함
    context = req.dict()
    context["input"] = context["user_message"]
    answer = multi_prompt_chain.run(context)

    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)