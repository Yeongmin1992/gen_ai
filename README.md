### gen_ai

practice for gen ai

패스트캠퍼스의 아래 강의들을 따라하며 진행한 실습입니다.

- ChatGPT API를 활용한 챗봇 서비스 구축 with LangChain & Semantic Kernel
- 알아서 일하는 진짜 인공지능 Auto-GPT 서비스 구현

### 1세대

유저의 요청이 APP으로 들어가서 prompt 1개 정도로만 사용하여 응답을 받는 방식

Langchain

- LLM
- PromptTemplate : 사용자의 질의에 맞추어서 응답을 변경할 수 있도록 함
- Chain : LLM과 PromptTemplate을 하나로 엮어서 chain을 실행하면 자동으로 user의 입력이
  PromptTemplate에 들어가사 prompt가 되고, prompt가 LLM에 넘어가서 결과값을 받게 됨.

### 2세대

prompt chaining을 통하거나 intent 기반으로 나눠서 다양한 상황에서 사용 가능

### LLM chain

- LLM
- Router
- SequentialChain : 각각의 chain을 묶어서 실행하는 하나의 chain

### 실행방법

Fast api는 python 파일 실행
Streamlit은 streamlit run app.py
