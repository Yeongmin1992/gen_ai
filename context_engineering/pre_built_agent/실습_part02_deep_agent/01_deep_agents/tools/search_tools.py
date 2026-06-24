"""웹 검색 도구 모듈.

Tavily API를 활용한 웹 검색 기능을 제공합니다.
"""

import json

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage


@tool(parse_docstring=True)
async def web_search(
    query: str,
    runtime: ToolRuntime,
) -> ToolMessage:
    """Search the web for information on a specific topic.

    This tool performs web searches and returns relevant results
    for the given query. Use this when you need to gather information from
    the internet about any topic.

    Args:
        query: The search query string. Be specific and clear about what
               information you're looking for.

    Returns:
        Search results from Tavily search engine.

    Example:
        web_search("machine learning applications in healthcare")
    """
    from tavily import AsyncTavilyClient

    client = AsyncTavilyClient()
    search_result = await client.search(query, search_depth="advanced", max_results=10)

    return ToolMessage(
        content=json.dumps(search_result, indent=2, ensure_ascii=False),
        tool_call_id=runtime.tool_call_id,
    )
