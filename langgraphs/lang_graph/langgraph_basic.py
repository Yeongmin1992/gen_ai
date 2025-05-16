# StateGraph로 상태 만들기

# 타입을 지정하여 dictionary 생성
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    counter: int
    alphabet: list[str]

graph_builder = StateGraph(State)

# Annotated로 특성 정보 까지 추가
from typing import Annotated
from typing_extensions import TypedDict
import operator

def node_a(state: State):
    state['counter'] += 1
    state['alphabet'] = ["Hello"]
    return state

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", node_a)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# 그래프를 실행 가능한 객체로 만들기
graph = graph_builder.compile()

from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# 초기 상태 정의
initial_state = {
    "counter": 0,
    "alphabet": []
}

state = initial_state

for _ in range(3):
    state = graph.invoke(state)
    print(state)

class State(TypedDict):
    counter: int
    # list 안의 string이 계속해서 더해질 수 있도록
    alphabet: Annotated[list[str], operator.add]

def node_a(state: State):
    state['counter'] += 1
    state['alphabet'] = ["Hello"]
    return state

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", node_a)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

# 초기 상태 정의
initial_state = {
    "counter": 0,
    "alphabet": []
}

state = initial_state

# Annotated로 인해 앞의 결과와 달리 Hello가 3개 list 안에 있음(그래프 내의 기억을 계속 남겨둘 때 사용. 그래프 내에서만 지속!)
for _ in range(3):
    state = graph.invoke(state)
    print(state)

# Message를 담는 StateGraph 만들기
from typing import Annotated

from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
# langgraph에서 위의 operator.add과 같은 기능을 모듈화해 둠
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

llm = ChatOpenAI(model="gpt-4o-mini")

# 챗봇의 대화 내용을 계속 챗봇으로 보내줌(기억하게 함)
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
# graph_builder.add_edge(START, "chatbot") 이것과 동일
graph_builder.set_entry_point("chatbot")
# graph_builder.add_edge("chatbot", END) 이것과 동일
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()

# 위의 그래프와 동일한 모양
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    # invoke를 하면 한번 보여지고 끝이나, for문과 stream 함수를 쓰면 대화가 한턴 한턴 다 보여진다.
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            # state안에 계속하여 메세지를 쌓으니 가장 최신의 메세지를 가져오겠다.
            print("Assistant:", value["messages"][-1].content)


from langchain_openai import ChatOpenAI
# 위의 State class선언과 add_message를 wrapping한 모듈
from langgraph.graph import MessagesState

graph_builder = StateGraph(MessagesState)

llm = ChatOpenAI(model="gpt-4o-mini")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
# graph_builder.add_edge(START, "chatbot") 이것과 동일
graph_builder.set_entry_point("chatbot")
# graph_builder.add_edge("chatbot", END) 이것과 동일
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    # invoke를 하면 한번 보여지고 끝이나, for문과 stream 함수를 쓰면 대화가 한턴 한턴 다 보여진다.
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            # state안에 계속하여 메세지를 쌓으니 가장 최신의 메세지를 가져오겠다.
            print("Assistant:", value["messages"][-1].content)

# State class선언과 add_message를 wrapping한 모듈에 counter라는 int 타입 속성도 추가해주기
class State(MessagesState):
    counter: int

graph_builder = StateGraph(State)

llm = ChatOpenAI(model="gpt-4o-mini")

def chatbot(state: State):
    state['counter'] = state.get('counter', 0) + 1
    return {
        "messages": [llm.invoke(state["messages"])],
        "counter": state['counter']
        }

graph_builder.add_node("chatbot", chatbot)
# graph_builder.add_edge(START, "chatbot") 이것과 동일
graph_builder.set_entry_point("chatbot")
# graph_builder.add_edge("chatbot", END) 이것과 동일
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()

from langchain_core.messages import HumanMessage

# 초기 상태 설정
initial_state = {
    "messages": [HumanMessage(content="Hello!")],
    "counter": 0
}

# 그래프 실행
result = graph.invoke(initial_state)

print(f"Final state: {result}")

state = initial_state

for _ in range(3):
    state = graph.invoke(state)
    print(f"Counter: {state['Counter']}")
    print(f"Last message: {state['messages'][-1].content}")
    print("---")