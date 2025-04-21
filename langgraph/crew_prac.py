from dotenv import load_dotenv
import os
import openai
from crewai import Agent, Task, Crew
from crewai.process import Process
from langchain_openai import ChatOpenAI
from IPython.display import display, Markdown

load_dotenv()

# print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
# pip install -q crewai crewai-tools


# 목차 설정 에이전트
outline_generator = Agent(
    role='Outline Generator',
    goal='Create structured outlines for articles on given topics. answer in Korean',
    llm = ChatOpenAI(model = "gpt-4o-mini", max_tokens=1000),
    backstory='You are an expert at organizing information and creating comprehensive outlines for varios subjects.'
)

# 본문 작성 에이전트
writer = Agent(
    role='Writer',
    goal='Create engaging content based on research. Answer in Korean',
    llm = ChatOpenAI(model = "gpt-4o", max_tokens=3000),
    backstory='You are a skilled writer who can transfor complex information into readable content.'
)

# Task 정의
outline_task = Task(
    description='Create a detailed outline for an article about AI\'s impoact on job markets',
    agent=outline_generator,
    expected_output="""
    A cmprehensive outline covering the main aspects of AI\'s influence on employment
    """
)

writing_task = Task(
    description='Write an article about the findings from the research',
    agent=writer,
    expected_output='An engaging article dscussing AI\'s influence on job markets'
)

# Crew 정의
ai_impact_crew = Crew(
    agents=[outline_generator, writer],
    tasks=[outline_task, writing_task],
    verbose=True
)

# Process 정의
# Process 정의가 없으면 agent들이 알아서 작업 수행 및 흐름 관리 > process로 컨트롤 가능
ai_impact_crew = Crew(
    agents=[outline_gnerator, writer],
    tasks=[outline_task, wrting_task],
    verbose=True,
    Process=Process.sequentail
)


if __name__ == "__main__":
    result = ai_impact_crew.kickoff()
    print(result)
    result = ai_impact_crew.kickoff()
    print(result)
    # 이렇게 하면 마크다운을 보기좋게 확인 가능
    display(Markdown(result.raw))