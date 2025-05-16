# langgraph를 사용하지 않고, tool 구현

from typing import Literal
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class State(TypedDict):
    counter: int
    alphabet: list[str]

def route_tools(
    state: State
) -> Literal["tools", "__end__"]:
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messags found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"

graph_builder = StateGraph(State)

llm = ChatOpenAI(model="gpt-4o-mini")

# 챗봇의 대화 내용을 계속 챗봇으로 보내줌(기억하게 함)
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# 답변 할 수 있음 직접 답변하고, 아닐 경우 tool calling
graph_builder.add_conditional_edges(
    "chatbot",    # 첫번 째 노드
    route_tools,   # 아래의 2개중 어디로 보낼지 결정
    {"tools": "tools", "__end__": "__end__"}
)


from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from langgraph.prebuilt import ToolNode

@tool
def get_weather(location: str):
    """Call to get the weather"""
    if location in ["서울", "인천"]:
        return "It's 60 degress and foggy."
    else:
        return "It's 90 degrees and suny."
    
@tool
def get_coolest_cities():
    """Get a list of collest cities"""
    return "서울, 고성"

tools = [get_weather, get_coolest_cities]
tool_node = ToolNode(tools)

from langchain_openai import ChatOpenAI

model_with_tools = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
).bind_tools(tools)

model_with_tools.invoke("서울 날씨는 어때?").tool_calls
model_with_tools.invoke("대한민국 대톨령이 누구야?").tool_calls
model_with_tools.invoke("서울 날씨는 어때?")
model_with_tools.invoke("한국에서 가장 추운 도시는?").tool_calls
tool_node.invoke({"messages": [model_with_tools.invoke("서울 날씨는 어떄?")]})
tool_node.invoke({"messages": [model_with_tools.invoke("한국에서 가장 추운 도시는?")]})