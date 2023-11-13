import json
import os
from uuid import uuid4

from dotenv import load_dotenv
import markdown
import nbformat
from bs4 import BeautifulSoup
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding
from semantic_kernel.connectors.memory.chroma import ChromaMemoryStore
from semantic_kernel.text.text_chunker import (
    split_markdown_paragraph,
    split_plaintext_paragraph
)

from dotenv import load_dotenv

load_dotenv()

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_PERSIST_DIR = os.path.join(CUR_DIR, "chroma-persist")
CHROMA_COLLECTION_NAME = "dosu-bot"

async def search_async(kernel, query):
    return await kernel.memory.search_async(
        collection=CHROMA_COLLECTION_NAME,
        query=query,
        limit=3,
        min_relevance_score=0    # 코사인 유사도. 경험에 따라 나오는 값으로 업로드된 데이터에 따라 달라짐.
    )
if __name__ == "__main__":
    kernel = Kernel()
    kernel.add_text_embedding_generation_service(
        "ada",
        OpenAITextEmbedding(
            "text-embedding-ada-002",
            os.getenv("OPENAI_API_KEY")
        )
    )
    kernel.register_memory_store(
        memory_store=ChromaMemoryStore(persist_directory=CHROMA_PERSIST_DIR)
    )


    import asyncio
    from pprint import pprint


    docs = asyncio.run(
        search_async(kernel, "I want to know about planner")
    )
    pprint([doc.text for doc in docs])