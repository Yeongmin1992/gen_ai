import os

import openai
import sys
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("openai_api_key")

print(sys.version)

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system", "content": "You are a helpful assistant,"
        },
        {
            "role": "user", "content": "What can you do?"
        }
    ],
    # 랜덤성 : 0 ~ 1 사이의 값으로 0에 가까울수록 고정된 답변을 반환하게 됨
    temperature = 0,
    # 응답을 몇개 받을 것인지
    n=1,
    # 응답 길이 제한(보통 영어의 경우 1토큰에 4글자)
    max_tokens=3500
    # 리스트 안에 있는 문자가 나올 경우 응답 중지
    # stop=[".", ","]
)

response = response.choices[0].message.content
print(response)