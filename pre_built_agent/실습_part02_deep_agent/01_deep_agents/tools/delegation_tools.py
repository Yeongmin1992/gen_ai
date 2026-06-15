"""Sub-agent 위임 도구 모듈.

Context Isolation 패턴을 구현하여 복잡한 작업을 독립된 Sub-agent에게 위임.
각 Sub-agent는 격리된 컨텍스트에서 작업을 수행하여 컨텍스트 충돌을 방지.
"""

from typing import NotRequired, TypedDict

from langchain.agents import create_agent
from langchain.tools import BaseTool, ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

# =============================================================================
# Sub-agent Configuration
# =============================================================================


class SubAgent(TypedDict):
    """Sub-agent 구성 정보.

    Attributes:
        name: 에이전트 식별자 (main agent에서 호출 시 사용)
        description: 역할 설명 (main agent에서 호출 시 사용)
        prompt: 전용 시스템 프롬프트 (sub-agent 작업 지시)
        tools: 사용 가능한 도구 이름 목록 (선택적)
    """

    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]


# =============================================================================
# Task Delegation Tool
# =============================================================================

TASK_DESCRIPTION_PREFIX = """Delegate a task to a specialized sub-agent with isolated context.

Use this tool to spawn a sub-agent that operates with a clean context window,
preventing context clash from the parent agent's conversation history.

**Available Sub-agents:**
{other_agents}

**Parameters:**
- description: Clear, specific task description (must be self-contained)
- subagent_type: Type of sub-agent to use

**Context Isolation:** The sub-agent receives ONLY the task description,
not the full conversation history. Provide complete context in the description.
"""


def create_task_tool(tools: list, subagents: list[SubAgent], model, state_schema):
    """Create a task delegation tool that enables context isolation through sub-agents.

    이 함수는 격리된 컨텍스트를 가진 특화 Sub-agent를 생성하는 핵심 패턴을 구현합니다.
    복잡한 멀티스텝 작업에서 컨텍스트 충돌과 혼란을 방지합니다.

    Args:
        tools: Sub-agent에 할당할 수 있는 도구 목록
        subagents: 특화된 Sub-agent 구성 정보 목록
        model: 모든 에이전트에 사용할 언어 모델
        state_schema: 상태 스키마 (일반적으로 DeepAgentState)

    Returns:
        특화된 Sub-agent에게 작업을 위임할 수 있는 'task' 도구
    """
    # Sub-agent 레지스트리 생성
    agents = {}

    # 도구 이름별 매핑 생성
    tools_by_name = {}
    for tool_ in tools:
        if not isinstance(tool_, BaseTool):
            tool_ = tool(tool_)
        tools_by_name[tool_.name] = tool_

    # Sub-agent 생성 및 레지스트리 등록
    for _agent in subagents:
        if "tools" in _agent:
            _tools = [tools_by_name[t] for t in _agent["tools"]]
        else:
            _tools = tools

        agents[_agent["name"]] = create_agent(
            model,
            system_prompt=_agent["prompt"],
            tools=_tools,
            state_schema=state_schema,
        )

    # Sub-agent 목록 문자열 생성
    other_agents_string = [
        f"- {_agent['name']}: {_agent['description']}" for _agent in subagents
    ]

    @tool(
        description=TASK_DESCRIPTION_PREFIX.format(
            other_agents="\n".join(other_agents_string)
        )
    )
    async def task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ):
        """Delegate a task to a specialized sub-agent with isolated context.

        This creates a fresh context for the sub-agent containing only the task description,
        preventing context pollution from the parent agent's conversation history.

        Args:
            description: Clear, specific task description (self-contained)
            subagent_type: Type of sub-agent to use
            runtime: Runtime context (injected)

        Returns:
            Command with sub-agent's results merged into parent state
        """
        if subagent_type not in agents:
            return f"Error: invoked agent of type {subagent_type}, the only allowed types are {list(agents.keys())}"

        sub_agent = agents[subagent_type]

        # 격리된 컨텍스트 생성 - 작업 설명만 포함
        isolation_state = {"messages": [{"role": "user", "content": description}]}

        # Sub-agent 실행
        result = await sub_agent.ainvoke(isolation_state)

        # 결과를 Command로 래핑하여 반환
        return Command(
            update={
                "files": result.get("files", {}),
                "messages": [
                    ToolMessage(
                        result["messages"][-1].content,
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    return task
