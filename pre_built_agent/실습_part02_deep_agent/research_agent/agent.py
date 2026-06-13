"""DeepAgent 기반의 리서치 에이전트."""

from datetime import datetime

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

from .prompts import (
    RESEARCH_WORKFLOW_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
)
from .tools import tavily_search, think_tool

# 한도 설정
max_concurrent_research_units = 3
max_researcher_iterations = 3

# 현재 날짜 계산
current_date = datetime.now().strftime("%Y-%m-%d")

# 오케스트레이터용 지침 결합 (RESEARCHER_INSTRUCTIONS 는 서브 에이전트에만 사용)
INSTRUCTIONS = (
    RESEARCH_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_research_units=max_concurrent_research_units,
        max_researcher_iterations=max_researcher_iterations,
    )
)

# 리서치 담당(Search, Reflection) - 서브 에이전트 생성
research_sub_agent = {
    "name": "research-agent",
    "description": "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=current_date),
    "tools": [tavily_search, think_tool],
}

model = ChatOpenAI(model="gpt-4.1", temperature=0.0)

# Deep Agent 생성
agent = create_deep_agent(
    model=model,
    tools=[tavily_search, think_tool],
    system_prompt=INSTRUCTIONS,
    subagents=[research_sub_agent],
)
