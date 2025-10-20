import warnings
warnings.filterwarnings('ignore')

import os

from dotenv import load_dotenv

# open ai key랑 tavily api key 필요
load_dotenv()

# State 설정
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]

# 도구 설정(웹 검색, PythonREPL)
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
# 코드가 주어졌을 때 로컬환경에서 실행하도록 도와줌
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool

web_search = TavilySearchResults(max_results=2)
repl = PythonREPL()

@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."]
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. chart labels should be written in English.
    This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Succesfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )
tools = [web_search, python_repl]
tool_node = ToolNode(tools)

# 에이전트에게 도구 인지시키기
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gtp-4o-mini")
llm_with_tools = llm.bind_tools(tools)

def agent(state: State):
    result = llm_with_tools.invoke(state["messages"])
    return {"messages": [result]}

def shold_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

workflow = StateGraph(State)

workflow.add_node("agent", agent)
workflow.add_node("tool", tool_node)

workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    "agent",
    shold_continue,
    {
        "continue": "tool",
        # Otherwise we finish,
        "end": END
    }
)

# We now add a normal edge from 'tools' to 'agent'.
# This means that after 'tool' is called, 'agent' node is called next.
workflow.add_edge("tool", "agent")

# Set up memory
memory = MemorySaver()

# tool node 실행 앞에서 멈추겠다. local 에서 repl 사용의 경우 사용자의 pc에 위협이 되는 코드를 생성 후 실행할 수 있기 때문.
graph = workflow.compile(checkpointer=memory, interrupt_before=["tools"])

from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))

import asyncio
initial_input = {"messages": [HumanMessage(content="미국의 최근 5개년(~2023) GDP 차트를 그려줄래?")]}
thread = {"configurable": {"thread_id": "13"}}
async def stream_async():
    async for chunk in graph.astream(initial_input, thread, stream_mode="updates"):
        for node, values in chunk.items():
            print(f"Receiving update from node: '{node}'")
            print(values)
            print("\n\n")
asyncio.run(stream_async())

# 위에서 thread_id를 주고 memory savor에 저장되도록 하였음으로, 위의 결과값을 가진채로 실행하게 된다.
async def stream_async():
    async for chunk in graph.astream(None, thread, stream_mode="updates"):
        for node, values in chunk.items():
            print(f"Receiving update from node: '{node}'")
            print(values)
            print("\n\n")
asyncio.run(stream_async())

async def stream_async():
    async for chunk in graph.astream(None, thread, stream_mode="updates"):
        for node, values in chunk.items():
            print(f"Receiving update from node: '{node}'")
            print(values)
            print("\n\n")
asyncio.run(stream_async())

async def stream_async():
    async for chunk in graph.astream(None, thread, stream_mode="updates"):
        for node, values in chunk.items():
            print(f"Receiving update from node: '{node}'")
            print(values)
            print("\n\n")
asyncio.run(stream_async())