import os
from typing import List
# langchain.embeddings 하위에 다양한 임베딩 방식 있음.
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

from dotenv import load_dotenv

load_dotenv()

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_PERSIST_DIR = os.path.join(CUR_DIR, "database", "chroma-persist")
CHROMA_COLLECTION_NAME = "dosu-bot"

_db = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=OpenAIEmbeddings(),
    collection_name=CHROMA_COLLECTION_NAME
)

_retrever = _db.as_retriever()

def query_db(query: str, use_retriever: bool = False) -> List[str]:
    if use_retriever:
        docs = _retrever.get_relevant_documents(query)
    else:
        docs = _db.similarity_search(query)

    str_docs = [doc.page_content for doc in docs]
    return str_docs