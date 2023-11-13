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
DATA_DIR = os.path.join(os.path.dirname(CUR_DIR), "dataset")

SK_CODE_DIR = os.path.join(DATA_DIR, "semantic-kernel", "python")
SK_SAMPLE_DIR = os.path.join(DATA_DIR, "semantic-kernel", "python", "notebooks")
SK_DOC_DIR = os.path.join(DATA_DIR, "semantic-kernel-docs", "semantic-kernel")

CHROMA_PERSIST_DIR = os.path.join(CUR_DIR, "chroma-persist")
CHROMA_COLLECTION_NAME = "dosu-bot"

LOADER_DICT = {
    "py": TextLoader,
    "md": UnstructuredMarkdownLoader,
    "ipynb": NotebookLoader,
}

def upload_embedding_from_file(file_path):
    loader = LOADER_DICT.get(file_path.split(".")[-1])
    if loader is None:
        raise ValueError("Not supported file type")
    
    documents = loader(file_path).load()
    # chunk size로 글자를 자르다보면 잘못 잘리는 경우가 있어 chunk overlap 설정으로 보완
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    Chroma.from_documents(docs, OpenAIEmbeddings(), collection_name=CHROMA_COLLECTION_NAME, persist_directory=CHROMA_PERSIST_DIR)

def upload_embeddings_from_dir(dir_path):
    failed_upload_files = []

    for root, dirs, files in os.walk(dir_path):
        print(files)
        for file in files:
            if file.endswith(".py") or file.endswith(".md") or file.endswith(".ipynb"):
                file_path = os.path.join(root, file)
                print(file_path)
                try:
                    upload_embedding_from_file(file_path)
                    print("SUCCESS: ", file_path)
                except Exception:
                    print("FAILED: ", file_path)
                    failed_upload_files.append(file_path)

if __name__ == "__main__":
    print("start")
    print(CUR_DIR)
    print(DATA_DIR)
    print(SK_CODE_DIR)
    upload_embeddings_from_dir(SK_CODE_DIR)
    upload_embeddings_from_dir(SK_SAMPLE_DIR)
    upload_embeddings_from_dir(SK_DOC_DIR)