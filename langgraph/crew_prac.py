from dotenv import load_dotenv
import os
import openai
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

load_dotenv()

# pip install -q crewai crewai-tools

openai.api_key = os.getenv("OPENAI_API_KEY")

# 목차 설정 에이전트
outline_generator = Agent(
    role='Outline Generator',
    goal='Create structured outlines for articles on given topics. andser in Korean',
    llm = ChatOpenAI(model = "gpt-4o-mioni", max_tokens=1000),
    backstory='You are an expert at organizing information and creating comprehensive outlines for varios subjects.'
)

# 본문 작성 에이전트
writer = Agent(
    role='Writer',
    goal='Create engaging content based on research. Answer in Korean'
    backstory='You are a skilled writer who can transfor complex information into readable content.'
)
