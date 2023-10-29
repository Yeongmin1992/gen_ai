from typing import List
import asyncio
from concurrent.futures import ProcessPoolExecutor
import datetime
import pynecone as pc
import requests
from bs4 import BeautifulSoup
import pandas as pd
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)

from langchain.schema import SystemMessage

from langchain.utilities import DuckDuckGoSearchAPIWrapper
import tiktoken

# chat completion 사용
def builder_summerizer(llm):
    system_message = "assistant는 user의 내용을 bullet point 3줄로 요약하라. 영어인 걍우 한국어로 번역해서 요약하라."
    system_message_prompt = SystemMessage(content=system_message)

    human_template = "{text}\n---\n위 내용을 bullet point 3줄로 한국어로 요약해"
    humman_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, humman_message_prompt])

    # llm과 prompt를 사용하면 summarize할 수 있는 체인이 됨
    chain = LLMChain(llm=llm, prompt=chat_prompt)
    return chain

def truncate_text(text, max_tokens=3000):
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])

def clean_html(url):
    """
    html의 텍스트 추출
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = ' '.join(soup.stripped_strings)
    return text

def task(search_result):
    title = search_result['title']
    url = search_result['link']
    snippet = search_result['snippet']

    content = clean_html(url)
    full_content = f"제목: {title}\n발췌: {snippet}\n전문: {content}"

    full_content_truncated = truncate_text(full_content, max_tokens=3500)

    summary = summarizer.run(text=full_content_truncated)

    result = {
        "title": title,
        "url": url,
        "content": content,
        "summary": summary
    }

    return result


##########################################################
# Instances
llm = ChatOpenAI(temperature=0.8)

# search.run("AI")
# search.results("AI", num_results=2)
search = DuckDuckGoSearchAPIWrapper()
search.region = 'kr-kr'

# enc.encode("오늘 날씨") 하면 숫자(토큰)의 리스트 형식 반환
# chat gpt 모델은 리스트 내의 요소 갯수인 토큰 수 만큼 과금되며
# 토큰 제한이 있음 
enc  = tiktoken.encoding_for_model("gpt-4")

# 요약
summarizer = builder_summerizer(llm)

##########################################################

class Data(pc.Model, table=True):
    """A table for questions and ansewrs in the database."""

    title: str
    content: str
    url: str
    summary: str
    timestamp: datetime.datetime = datetime.datetime.now()

class State(pc.State):
    is_working: bool = False,
    columns: List[str] = ["title", "url", "summary"]
    # State의 인스턴스 변수를 생성하면 자동으로 set_topic과 같은 수정자가 생성됨
    topic: str = ""

    async def handle_submit(self):
        self.is_working = True
        yield

        topic = self.topic

        search_results = search.results(topic, num_results=3)

        with ProcessPoolExecutor() as executor:
            with pc.session() as session:
                for s in search_results:
                    s = await asyncio.get_running_loop().run_in_executor(executor, task, s)
                    record = Data(
                        title=s['title'],
                        content=s['content'],
                        url=s['url'],
                        summary=s['summary']
                    )
                    # db에 레코드 하나 넣게 됨
                    session.add(record)
                    # db에 add한 것이 반영됨
                    session.commit()
                    # 화면 엡데이트를 위해 yield
                    yield

        self.is_working = False

@pc.var
def data(self)

def index() -> pc.Component:
    return pc.center(
        pc.vstack(
            pc.heading("뉴스 크롤링 & 요약 서비스", font_size="2em"),
            pc.input(placeholder="topic", on_blur=State.set_topic),
            pc.hstack(
                pc.button("시작", on_click=State.handle_submit),
                pc.button("excel export", on_click=State.export),
                pc.button("모두 삭제", on_click=State.delete_all)
            ),
            pc.cond(
                State.is_working,
                pc.spinner(
                    color="lightgreen",
                    thickness=5,
                    speed="1.5s",
                    size="xl"
                )
            ),
            pc.data_table(
                data=State.data,
                columns=State.columns,
                pagination=True,
                search=True,
                sort=False
            ),
            width="80%",
            font_size="1em",
        ),
        padding_top="10%"
    )