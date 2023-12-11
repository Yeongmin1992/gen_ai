import os
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI
from chains import (
    bug_step1_chain,
    bug_step2_chain,
    default_chain,
    enhance_step1_chain,
    parse_intent_chains,
    read_prompt_template
)
from database import query_db
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

# fastapi post 요청시 request를 편리하게 담아서 사용하기 위해 BaseModel 사용
class UserRequest(BaseModel):
    user_message: str

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
INTENT_LIST_TXT = os.path.join(CUR_DIR, "prompt_templates", "intent_list.txt")

@app.post("/qna")
def generate_answer(req: UserRequest) -> Dict[str, str]:
    # multipromptchain은 반드시 input키로 넣어줘야 함
    context = req.dict()
    context["input"] = context["user_message"]
    context["intent_list"] = read_prompt_template(INTENT_LIST_TXT)
    print(context)
    # parse_intent_chains의 output key를 intent로 했기 때문에 아래와 같이 dictionary에서 intent를 가져오는 방법과
    intent = parse_intent_chains(context)["intent"]

    # string으로 실행하는 방법
    #intent = parse_intent_chains.run(context)
    print("intent is: ")
    print(intent)

    if intent == "bug":
        # 답변하기 전 db에서 값을 가져와서 context에 추가
        context["related_documents"] = query_db(context["user_message"])

        answer = ""
        for step in [bug_step1_chain, bug_step2_chain]:
            # bug_step2_chain을 실행하기 전 bug_analysis를 담아준다.
            context = step(context)
            answer += context[step.output_key]
            answer += "\n\n"
    elif intent == "enhancement":
        answer = enhance_step1_chain.run(context)
    else:
        # default_chain은 ConversationChain을 타도록 되어있는데 input을 하나 밖에 받을 수 없음!
        answer = default_chain.run(context["user_message"])

    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)