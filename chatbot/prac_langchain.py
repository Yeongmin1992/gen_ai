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

from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    {   
        # SystemMessage: 유용한 챗봇이라는 역할과 이름 부여
        ("system", "You are a helpful AI bot. Your name is {name}."),
        # HumamMessage와 AIMessage: 서로 안부를 묻고 답하는 대화 히스토리 주입
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        # HumanMessage로 사용자가 입력한 프롬프트 전달
        ("human", "{user_input}")
    }
)

messages = chat_template.format_messages(name="Bob", user_input="What is your name?")
print(messages)

# LCEL(Lang Chain Expression Language)로 Chain 구축하기
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 프롴프트 템플릿 설정
prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")

# LLM 호출
model = ChatOpenAI(model="gpt-4o-mini")

# LCEL로 프롬프트 템플릿 > LLM > 츨력 파서 연결하기
# StrOutputParser를 사용하면 위의 AIMessage 객체에서 나온 다양한 값중 content 값만 string으로 보기좋게 나옴
chain = prompt | model | StrOutputParser()

# invoke 함수로 chain 실행하기
chain.invoke({"tokpic": "ice cream"})

# chain 선언
model = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

# Chain의 stream() 함수를 통해 스트리밍 기능 추가
for s in chain.stream({"topic":"bears"}):
    print(s.content, end="", flush=True)

# OutputParser 실습
from langchain_openai import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0
)

# ChatPromptTemplate에 SystemMessage로 LLM의 역할과 출력 형식 지정
# output parser 없이 답변 형식을 지정 할 경우
chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "너는 영화 전문가 AI야. 사용자가 원하는 장르의 영화를 리스트 형태로 추천해줘."
                'ec) Query: SF 영화 3개 추천해줘 / 답변 : ["인터스텔라", "스페이스오디세이", "혹성탈출"]'
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}")
    ]
)
model = ChatOpenAI(model="gpt-40-mini")
chain = chat_template | model
chain.invoke("액션")

# output parser를 사용하여 답변 형식을 지정 할 경우
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate

# CSV 파서 선언
output_parser = CommaSeparatedListOutputParser()

# CSV 파서 작동을 위한 형식 지정 프롬프트 로드
format_instructions = output_parser.get_format_instructions()
# 답변 타입이 list of comma separated values라고 알아서 시스템메세지를 잘 만들어줌
print(format_instructions)

# 프롬프트 템플릿의 partial_variables에 CSV 형식 지정 프롬프트 주입
prompt = PromptTemplate(
    template="List {subject}. Answer in Korean \n{format_instructions}",
    input_variables=["subject"],
    # 요런 식으로 사용자 input 받기전 프롬프트 템플릿에 부분 변수로 넣기 가능
    partial_variables={"format_instuction": format_instructions}
)

model = ChatOpenAI(model="gpt-4o-mini")

# 프롬프트템플릿-모델-Output Parser를 체인으로 연결
chain = prompt | model | output_parser
chain.invoke({"subject": "공포 영화"})

from typing import List
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# pydantic_object의 데이터 구조를 정의합니다.
class Country(BaseModel):
    continent: str = Field(description="사용자가 물어본 나라가 속한 대륙")
    population: str = Field(description="사용자가 물어본 나라의 인구(int 형식)")

# JsonOutputParser를 설정하고, 프롬프트 템플릿에 format_instructions를 삽입합니다.
parser = JsonOutputParser(pydantic_object=Country)

prompt = PromptTemplate(
    template="Answer the user query. \n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | model | parser

country_query = "아르헨티나는 어떤 나라야?"
chain.invoke({"query":country_query})

# LCEL Runnable 객체에 대해 알아보기 > 파이프라인 오프레이터 좌우에 있는 태스크들은 runnable 객체이다.

# 들어온 객체를 그대로 내보내는 RunnablePassthrough
from langchain_core.runnables import RunnablePassthrough

RunnablePassthrough().invoke("안녕하세요")

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    """
    다음 한글 문장을 프랑스어로 번역해줘
    {sentence}
    French Sentencd: (print from here)
    """
)

# 사용자로 부터 입력받은 인풋을 그대로 넘김
runnable_chain = {"sentence": RunnablePassthrough()} | prompt | model | output_parser
runnable_chain.invoke({"sentence": "그녀는 매일 아침 책을 읽습니다."})

# assign 함수를 사용하여 들어온 input에 함수를 적용할 수 있음
(RunnablePassthrough.assign(mult=lambda x: x["num"]*3)).invoke({"num":3})

# 로직 병렬실행
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

runnable = RunnableParallel(
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"] * 3),
    modified=lambda x: x["num"] + 1
)

# RunnablePassthrough로 extra는 {'num': 1, 'mult' : 3}
# modified는 {'modified': 2} 와 같이 나옴
runnable.invoke({"num": 1})

# 사용자 정의 함수에 Runnable 객체 넣기
def add_smile(x):
    return x + ":)"

from langchain_core.runnables import RunnableLambda

add_smile = RunnableLambda(add_smile)

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

prompt_str = "{topic}의 역사에 대해 세문장으로 설명해주세요."
prompt = ChatPromptTemplate.from_template(prompt_str)

model = ChatOpenAI(model_name="gpt-4o-mini")

output_parser = StrOutputParser()

chain = prompt | model | output_parser

from langchain_core.runnables import RunnableLambda

def add_thank(x):
    return x + " 들어주셔서 감사합니다 :)"

add_thank = RunnableLambda(add_thank)

chain = prompt | model | output_parser | add_thank
chain.invoke("반도체")

# RunnableParallel 알아보기
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

runnable = RunnableParallel(
    passed=RunnablePassthrough(),
    modified=lambda x: x["num"] + 1
)

runnable.invoke({"num": 1})

runnable = RunnableParallel(
    passed=RunnablePassthrough(),
    modified=add_thank
)

runnable.invoke("안녕하세요.")

# RunnableParallel으로 여러 결과물 한번에 받기
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

model = ChatOpenAI(model = 'gpt-4p-mini', max_tokens = 128, temperature = 0)

history_prompt = ChatPromptTemplate.from_template("{topic}가 무엇의 약자인지 알려주세요.")
celeb_prompt = ChatPromptTemplate.from_template("{topic} 분야의 유명인사 3명의 이름만 알려주세요.")

output_parser = StrOutputParser()

history_chain = history_prompt | model | output_parser
celeb_chain = celeb_prompt | model | output_parser

map_chain = RunnableParallel(history=history_chain, celeb=celeb_chain)

result = map_chain.invoke({"topic":"AI"})
print(result)