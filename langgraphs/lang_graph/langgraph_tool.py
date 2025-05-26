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

from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, MessagesState

# 이전의 conditional edge와 동일한 역할
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 모델이 실행하는 결과의 output이 어떻게 나와야 하는지 정의
def call_model(state: MessagesState):
    messages = state['messages']
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")

# add a conditional edge
workflow.add_conditional_edges(
    "agent",
    should_continue
)

# tool에서 반드시 agent로 응답을 넘겨주는 부분
# 아래에서 가장 추운도시의 날씨는 어때? 라는 질문을 했을 때,
# 서울, 고성을 응답하고, agent로 다시 넘기면 서울은 60도, 고성은 90도라고 해 줌
workflow.add_adge("tools", "agent")

app = workflow.compile()

from IPython.display import Image, display

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

final_state = app.invoke(
    {"messages": [HumanMessage(content="서울의 날씨는 어떄?")]}
)
final_state["messages"][-1].content

# example with a multiple tool calls in succession
for chunk in app.stream(
    {"messages": [("human", "가장 추운 도시의 날씨는 어떄?")]},
    stream_mode="values"
):
    chunk["messages"][-1].pretty_print()


# 웹검색 tool 결합
from langgraph.graph.message import add_messages
# Annotated는 타입에 메타데이터를 추가하고자 할 때 사용. add_messages 함수값이 메타데이터임
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ToolNode로 도구 노드 구축
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition

tool = TavilySearchResults(max_results=2)
tools = [tool]
tool_node = ToolNode(tools)

# ToolNode 구조

import json

from langchain_core.messages import ToolMessage

# tool list가 있을 때, 사용자 질문을 받으면 ai가 tool에게 보낼 tool calls 라는 메세지 형태로 보내게 되고,
# 그것을 list 내의 도구들이 각각 실행하게 됨
class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                )
            )
        return {"messages": outputs}
    
# LLM 챗봇 설정
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    result = llm_with_tools.invoke(state["messages"])
    return {"messages": [result]}

# 그래프 구축
from langgraph.graph import StateGraph

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

# tools가 실행 된 경우엔 반드시 chatbot에 응답을 전달하도록
graph_builder.add_edge("tools", "chatbot")
# laggraph.prebuilt의 tools_condition 사용
graph_builder.add_conditional_edges("chatbot", tools_condition)

# 사용자의 질문이 들어오면 곧바로 chatbot 노드가 전달받을 수 있도록
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()

# 인터넷 검색이 필요한 질문
graph.invoke({"messages": {"role": "user", "content": "지금 한국 대통령은 누구야?"}})

# LLM이 답할 수 있는 질문
graph.invoke({"messages": {"role": "user", "content": "마이크로소프타가 어떤 회사야?"}})

# 기억력 만들기 im memory를 활용한 MemorySaver 외로 MysqlMemeorySaver 등 다양한memory saver 사용 가능
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
# 특정 쓰레드에 한해서 대화의 기록이 유지
graph = graph_builder.compile(checkpointer=memory)

# 대화의 쓰레드 id 지정 > 쓰레드를 유지하는한 대화를 기억. 대화 나가면 기억 못 함
config = {"configurable": {"thread_id": "1"}}
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages": ("user", user_input)}, config):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# 대화 내용이 어떻게 저장되어 있는지 확인
snapshot = graph.get_state(config)
print(snapshot)

# 쓰레드 바꾸면 기억 못 함
config2 = {"configurable": {"thread_id": "2"}}
graph.invoke({"messages": [{"role": "user", "content": "내가 한 첫 질문이 뭐였어?"}]}, config2)


# 기억할 메세지 개수 제한 하기 : context window의 제한이 있어 너무 많은 대화를 기억하면 에러가 날 수 있다.
def filter_messages(messages: list):
    # This is very simple helper function which only ever uses the last 2 messages
    return messages[-2:]

def chatbot(state: State):
    messages = filter_messages(state["messages"])
    result = llm_with_tools.invoke(messages)
    return {"messages": [result]}

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)

graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "20"}}
input_message = HumanMessage(content="Hi! I'm bob and I like soccer")
for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

# This will now not remember the previous messages
# (because we set messages[-1] in the filter messages argument)
input_message = HumanMessage(content="What's my name?")
for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

input_message = HumanMessage(content="What's my name?")
for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

# 마지막 대화를 기억하기 때문에 soccer를 좋아한다는 대화를 잊는다.
input_message = HumanMessage(content="What's my favorite?")
for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

# 필터 방식을 바꿔 요약된 대화 내용을 챗봇이 기억하게 하는 등으로 활용 가능

# Human in the loop를 통해 직접 개입하기
# interrunpt_before는 특정한 노드를 실행하기 전에 멈추는 옵션
graph = graph_builder.compile(checkpointer=memory, interrupt_before=["tools"])

user_input = "Langgraph가 뭐야?"
config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)

# 웹검색까지는 하지만, tool이 실행되는 것은 볼 수 없음
# 멈추고 사용자가 input을 넣어준다던지 이전의 state값을 수정하여 전달하는 등으로 활용 가능
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()