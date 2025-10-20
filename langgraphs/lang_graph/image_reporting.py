# multi agetn for image reporting
# pip install python-docx

import warnings
warnings.filterwarnings('ignore')

import os

from dotenv import load_dotenv

# open ai key랑 tavily api key 필요
load_dotenv()

# State 설정
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage
