import os
from langchain.document_loaders import (
    NotebookLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
# langchain.embeddings 하위에 다양한 임베딩 방식 있음.
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

from dotenv import load_dotenv

load_dotenv()

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_PERSIST_DIR = os.path.join(CUR_DIR, "chroma-persist")
CHROMA_COLLECTION_NAME = "dosu-bot"

if __name__ == "__main__":
    db = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME
    )

    # docs의 경우 Chroma db를 그대로 쓸 수가 있고, retriever로 바꿔서 쓸 수 가 있는데 Chroma db 그대로 사용을 추천

    docs = db.similarity_search("I want to know about planner")
    
    from pprint import pprint

    # pprint(docs)

    # retriever로 바꿔서 쓰기
    retriever = db.as_retriever()
    retriever.get_relevant_documents("I want to know about planner")

    pprint(docs)