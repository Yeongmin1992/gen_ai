import os

from dotenv import load_dotenv
# langchain 수학 계산, 웹 서핑
from langchain import LLMMathChain, SerpAPIWrapper
# 위의 두가지가 langchain의 tool이라는 것을 알려주기 위해 tool 임포트
from langchain.agents.tools import Tool
from langchain.chat_models import ChatOpenAI
# https://blog.langchain.dev/goodbye-cves-hello-langchain_experimental/
from langchain_experimental.plan_and_execute.executors import agent_executor
from langchain_experimental.plan_and_execute.planners import chat_planner
from langchain_experimental.plan_and_execute import PlanAndExecute
from langchain.llms import OpenAI
import openai

load_dotenv()

# https://serpapi.com/manage-api-key
# pip install google-search-results or poetry add google-search-results
llm = OpenAI(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),
]

model = ChatOpenAI(temperature=0)
planner = chat_planner(model)
executor = agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

agent.run(
    "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"
)