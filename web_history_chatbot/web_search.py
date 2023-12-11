import os
from chains import search_compression_chain, search_value_check_chain
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain.utilities.google_search import GoogleSearchAPIWrapper

load_dotenv()

search = GoogleSearchAPIWrapper(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    google_cse_id=os.getenv("GOOGLE_CSE_ID")
)

search_tool = Tool(
    name="Google Search",
    description="Search Google for recent results.",
    func=search.run
)

def query_web_search(user_message: str) -> str:
    context = {"user_message": user_message}
    context["related_web_search_results"] = search_tool.run(user_message)

    has_value = search_value_check_chain.run(context)

    print(has_value)
    # value가 있다면(검색 결과로 응답할만 한 값이 나온다면) compress 해서 값을 return하고, 아무것도 없을 경우 빈값 리턴
    if has_value == "Y":
        return search_compression_chain.run(context)
    else:
        return ""
    

# 예시 실행
# print(query_web_search("Obama's first name?"))