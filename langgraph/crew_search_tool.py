from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings('ignore')
from crewai import Agent, Task, Crew
from crewai.process import Process
from crewai_tools import (
    SerperDevTool,
    WebsiteSearchTool,
    ScrapeWebsiteTool
)

load_dotenv()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model = "gpt-4o-mini")

# 특정 키워드로 구글에 웹검색한 결과를 가져오는 도구
search_tool = SerperDevTool()
# 특정 키워드와 웹사이트를 주면 웹사이트에서 키워드와 관련된 결과만 주는 도구
web_rag_tool = WebsiteSearchTool()
# 웹사이트의 텍스트를 scraping
scrap_tool = ScrapeWebsiteTool()

researcher = Agent(
    role='테크 트렌드 연구원',
    goal="인공 지능 분야의 최신 기술 트렌드를 한국어로 제공합니다. 지금은 2025년 4월입니다.",
    # role과 goal에 작성하지 못한 추가적인 내용 기재
    backstory='기술 트렌드에 예리한 안목을 지닌 전문 분석가지아 AI 개발자입니다.',
    tools=[search_tool, web_rag_tool],
    verbose=True,
    # 주어진 태스크를 수행할 때 반복 수행 횟수를 제한
    max_iter=5,
    llm = llm
)

writer = Agent(
    role='뉴스레터 작성자',
    goal="최신 AI 기술 트렌드에 대한 매력적인 테크 뉴스레터를 한국어로 작성하세요. 지금은 2025년 4월입니다.",
    # role과 goal에 작성하지 못한 추가적인 내용 기재
    backstory='기술에 대한 열정을 가진 숙련된 작가입니다.',
    verbose=True,
    # 주어진 태스크를 수행할 때 다른 agent에게 태스크를 위임
    allow_delegation=False,
    llm = llm    
)

# Define tasks
research = Task(
    description='AI 업계 최신 기술 동향을 조사하고 요약을 제공하세요.',
    expected_output='AI 업계에서 가장 주목받는 3대 기술 개발 동향과 그 중요성에 대한 신선한 관점을 요약한 글',
    agent=researcher
)

write = Task(
    description="""
    테크 트렌드 연구원의 요약을 바탕으로 AI 산업에 대한 매력적인 테크 뉴스레터를 작성하세요.
    테크 뉴스레터이므로 전문적인 용어를 사용해도 괜찮습니다.""",
    expected_output="최신 기술 관련 소식을 재밌는 말투로 소개하는 4문단짜리 마크다운 형식 뉴스레터",
    agent=writer,
    output_filter=r'/mnt/d/ym/new_post.md'  # The final blog post wiil be saved here 
)

# Asseble a crew with planning enabled
crew = Crew(
    agents=[researcher, writer],
    tasks=[research, write],
    verbose=True,
    # crew가 직접 작업순서를 판단하게 할 수 도 있으나, sequential로 연구를 한 후 작성을하는 순서를 따르게 한다.
    process=Process.sequential
)

# Execute tasks
result = crew.kickoff()