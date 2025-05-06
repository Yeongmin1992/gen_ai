# langchain 모델 컴포ㅌ넌트 실습
import os
from dotenv import load_dotenv

load_dotenv()
from openai import OpenAI

# lamgchain 없이 쓰기
client = OpenAI()
# ChatCompletion 객체
print(client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "2002년 월드컵 4강 국가 알려줘"
        }
    ]
))

from langchain_openai import ChatOpenAI
# AIMessage 객체
chat  = ChatOpenAI(
    model_name = 'gpt-4o-mini'
)
print(chat.invoke("안녕~ 너를 소개해줄래?"))

from langchain.prompts import PromptTemplate

prompt = (
    PromptTemplate.from_template(
        """
        너는 요리사야. 내가 가진 재료들을 갖고 만들 수 있는 요리를 {개수} 추천하고,
        그 요리의 레시피를 제시해줘. 내가 가진 재료는 아래와 같아.
        <재료>
        {재료}
        """
    )
)
# PromptTemplate 객체
print(prompt)

# StringPromptValue 객체
print(prompt.invoke({"개수":6, "재료":"사과, 잼"}))