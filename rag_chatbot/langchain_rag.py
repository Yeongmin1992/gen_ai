# pip install -q pypdf faiss-cpu

import os
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"/content/auto_gen.pdf")

# pdf 파일 로드 및 페이지 별로 자르기
pages = loader.load_and_split()

# 웹페이지의 텍스트를 불러오기
from langchain_community.document_loaders import WebBaseLoader
# 텍스트 추출할 url 입력
loader = WebBaseLoader("https://www.espn.com/")
# ssl verification 에러 방지
loader.requests_kwargs = {'verify':False}
data = loader.load()
# html tag 베이스로 어느정도 전처리 해줌
print(data)

# 클래스 지정하여 특정 부분만 가져오기
import bs4
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://ww.espn.com/",
                       bs_kwargs=dict(
                           parse_only=bs4.SoupStrainer(
                               class_=("headlineStack top-headlines")
                           )
                       ))

loader.requests_kwargs = {'verify':False}
data = loader.load()
# html tag 베이스로 어느정도 전처리 해줌
print(data)

# text splitter로 문서 분할하기
loader = PyPDFLoader(r"/content/a.pdf")

# pdf 파일 로드 및 페이지 별로 자르기
pages = loader.load_and_split()

from langchain_text_splitters import CharacterTextSplitter

# 구분자: 줄넘김, 청크 길이: 500, 청크 오버랩: 100, length_function: 글자수
text_splitter = CharacterTextSplitter(
    # sperator="\n",
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
)

# 텍스트 분할
texts = text_splitter.split_documents(pages)
print(texts[0])
# 위에서 chunk size를 500을 주었음에도 2000 ~ 5000의 글자로 split됨(chunk가 제대로 적용되지 않음)
# separator 주석을 풀면 대부분 400 ~ 500글자로 chunking 잘 됨(seprator 단위로 최대한 chunk_size를 맞추려는 방식이라,
# 문단 길이가 chunk_size보다 크면 문단 전체를 가져다 씀)
print([len(i.page_content) for i in texts])

# RecursiveCharactorTextSpliter 사용하면 seprator 없이도 지정된 chunk_size로 chunking을 잘 함
# 여러 구분자(문단 > 문장 > 마침표 > 띄어쓰기)들을 재귀적으로 돌면서 chunk를 나누기 때문
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 구분자: 줄넘김, 청크 길이: 500, 청크 오버랩: 100, length_function: 글자수
text_splitter = RecursiveCharacterTextSplitter(
    # sperator="\n",
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
)

texts = text_splitter.split_documents(pages)
print([len(i.page_content) for i in texts])

# 임베딩 모델로 텍스트 수치화하기
from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(model = 'text-embedding-3-small')
embeddings = embeddings_model.aembed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)

# Hi There! 라는 글자를 1536 차원(숫자)로 임베딩함
len(embeddings), len(embeddings[0])

print(embeddings[0][:10])
print("-"*50)
print(f"임베딩 갯수: len(embeddings) \n임베딩 차원: len(embeddings[0])")


# chunking 후 embedding
# texts를 그대로 쓰면 Documents 객체의 메타데이터 등 다양한 데이터가 같이 들어가게 되어 page_content로 필요한 content 정보만 임베딩
embeddings = embeddings_model.embed_documents([i.page_content for i in texts])
len(embeddings), len(embeddings[0])

# 벡터 스토어에 저장

from langchain.vectorstores import FAISS

db = FAISS.from_documents(texts, embeddings_model)

# Retriever 생성
retriever = db.as_retriever()
query = "autogen이 뭐야?"
# 유사문서 검색
retriever.invoke(query)

from langchain_openai import ChatOpenAI
# langchain 커뮤니티에서 프롬프트를 공유하는 커뮤니티
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

model = ChatOpenAI(model="gpt-4o-mini")
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

rag_chain.invoke("autogen이 뭐야?")