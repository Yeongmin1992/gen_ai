"""TODO 리스트 관리 도구 모듈.

에이전트가 복잡한 워크플로우에서 작업 진행 상황을 추적하고 관리하기 위한 도구들.
- write_todos: TODO 리스트 생성 및 업데이트
- read_todos: 현재 TODO 리스트 조회
"""

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

# =============================================================================
# Tool Descriptions
# =============================================================================

WRITE_TODOS_DESCRIPTION = """Create and manage a structured task list for tracking progress through complex workflows.

**When to use**:
- Multi-step or complex tasks requiring coordination
- When user provides multiple tasks or explicitly requests a TODO list
- Avoid for simple tasks unless instructed otherwise

**Structure**:
- Maintain a single list containing multiple TODO objects (content, status)
- Use clear, actionable content descriptions
- Status values: pending, in_progress, completed

**Best Practices**:
- Only one task should be in_progress at a time
- Mark tasks completed immediately upon finishing
- Always send the complete updated list when making changes
- Remove irrelevant items to keep list focused

**Progress Updates**:
- Re-invoke TodoWrite whenever task status changes or content needs modification
- Reflect real-time progress, don't batch completions
- If blocked, keep task in_progress and add new task describing the blocker

**Parameters**:
- todos: List of TODO items with content and status fields

**Returns**:
- Updates agent state with new TODO list
"""


# =============================================================================
# Tool Implementations
# =============================================================================


@tool(description=WRITE_TODOS_DESCRIPTION, parse_docstring=True)
async def write_todos(
    todos: list[dict],
    runtime: ToolRuntime,
) -> Command:
    """Create or update the agent's TODO list for task planning and tracking.

    Args:
        todos: List of Todo items with content and status
        runtime: Runtime context (injected)

    Returns:
        Command to update agent state with new TODO list
    """
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(
                    f"Updated todo list to {todos}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool(parse_docstring=True)
async def read_todos(
    runtime: ToolRuntime,
) -> str:
    """Read the current TODO list from the agent state.

    This tool allows the agent to retrieve and review the current TODO list
    to stay focused on remaining tasks and track progress through complex workflows.

    Args:
        runtime: Runtime context (injected)

    Returns:
        Formatted string representation of the current TODO list
    """
    todos = runtime.state.get("todos", [])
    if not todos:
        return "현재 TODO 리스트가 비어 있습니다."

    status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}

    result = "현재 TODO List:\n"
    for i, todo in enumerate(todos, 1):
        emoji = status_emoji.get(todo["status"], "❓")
        result += f"{i}. {emoji} {todo['content']} ({todo['status']})\n"

    return result.strip()
