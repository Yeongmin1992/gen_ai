"""
기존 RAG(Naive RAG)의 문제점

1. 사용자의 질문이 인덱싱된 문서와 관련이 없는 경우에도 Retrieval 단계를 거침
2. 검색된 문서가 사용자의 질문과 일정 수준 이상 관련성이 없는 경우에도 이를 활용해 답변하려고 하기 때문에 환각 현상 크게 증가


Agentic Rag

2개의 분기(검색 필요 여부, 관련성 검토)로 사용자의 질문에 더 제대로된 답변을 할 수 있음
인덱싱되지 않은 문서는 검색을 수행하지 않거나, 관련된 문서를 찾을 수 없는 경우, 쿼리를 재작성 하는 등의 과정 수행
"""

from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
print(docs_list)

# Chrome 벡터 DB에 임베딩 저장하기
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)

doc_splits = text_splitter.split_documents(docs_list)
chroma = Chroma()
# Add to vectorDB
vectorstore = chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings
)
retriever = vectorstore.as_retriever()

# Retriever를 도구로 저장
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",      # retriever 이름
    "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs."     # retriever 설명 
)

tools = [retriever_tool]

# AgentState 선언하기
from typing import Annotated, Literal, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 문서 관련성 검토 함수 정의하기
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import tools_condition

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relavant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: Adecision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary scores for relevance check."""
        binary_score: str = Field(description="Relavance score 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)
    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader asssing relevence of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevnt to the question.""",
        input_variables=["context", "question"]
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_massage = messages[-1]

    question = messages[0].content
    docs = last_massage.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    # generate node로 가도록
    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"
    # 재작성 node로 가도록
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"
    
# 사용자와 상호작용하는 에이전트 함수 정의하기
def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): the current state
    
    Returns:
        dict: The updated state with the agents response appended to messages
    """
    print("---CALL AGNET---")
    messages = state["messages"]
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-40-mini")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We reuturn a list, because this well get added to the existing list
    return {"messages": [response]}

def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f"""\n
            Look at the input and try to reason about the underlying semantic intent / meaning. \n
            Here is the initial question:
            \n ------- \n
            {question}
            \n ------- \n
            Formulate an imporoved question.
            """
        )
    ]

    # Grader
    model = ChatOpenAI(temperature=0, model="gpt-40-mini", streaming=True)
    response = model.invoke(msg)
    return {"messages": [response]}

# 답변 함수 정의하기
def generate(state):
    """
    Generator answer

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": {response}}

print("*" * 20 + "Prompt[rlm/rag-prompt]" + "*" * 20)
prompt = hub.pull("rlm/rag-prompt").pretty_print()  # Show what the prompt looks like

# 그래프 구축하기
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)   # agent
retrieve =  ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)     # retrieval
workflow.add_node("rewrite", rewrite)   # Re-writing the question
workflow.add_node("generate", generate)     # Generating a response after we know the documents are relevant

# Call agent node to decide to retreive or not
workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "retreive",
        END: END
    }
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile
graph = workflow.compile()

# 그래프 시각화
from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies tand is optional
    pass

# 그래프 실행해보기
query_1 = "agent memory가 무엇인가요?"
# 블로그에 없는 내용으로 rewrite을 호출 해야 함
query_2 = "Lilian Weng은 agent memeory를 어떤 것에 비유했나요?"

import pprint
inputs = {"messages": [("user", query_1)]}

# 뒤에 있는 dictionary는 config로 graph 세팅 값. recursion_limit이 없을 경우 관련되지 않은 질문이 들어올 시 무한 루프 돌기 때문에 횟수 제한
for output in graph.stream(inputs, {"recursion_limit":10}):
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
    print("\n---\n")