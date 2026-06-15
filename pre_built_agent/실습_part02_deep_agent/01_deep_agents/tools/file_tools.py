"""가상 파일 시스템 도구 모듈.

LangGraph State 내에서 동작하는 가상 파일 시스템(Virtual File System) 도구들.
Context Offloading 패턴을 구현하여 컨텍스트 윈도우를 효율적으로 관리.

동작 방식: `Command` 타입을 활용한 Agent State 업데이트

- ls: 파일 목록 조회
- read_file: 파일 내용 읽기 (Pagination 지원)
- write_file: 파일 생성 및 덮어쓰기
"""

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

# =============================================================================
# Tool Descriptions
# =============================================================================

LS_DESCRIPTION = """List all files in the virtual filesystem stored in agent state.

Use this to check what files currently exist in agent memory before
performing other file operations. Helpful for understanding available files.

No parameters required - just call ls() to see all available files."""

READ_FILE_DESCRIPTION = """Read content from a file in the virtual filesystem with optional pagination.

This tool returns file content with line numbers (like `cat -n`) and supports
reading large files in chunks to avoid context overflow.

Parameters:
- file_path (required): Path to the file you want to read
- offset (optional, default=0): Line number to start reading from
- limit (optional, default=2000): Maximum number of lines to read

Essential before making any edits to understand existing content.
Always read a file before editing it."""

WRITE_FILE_DESCRIPTION = """Create a new file or completely overwrite an existing file in the virtual filesystem.

This tool creates new files or replaces entire file contents. Use for initial
file creation or complete rewrites. Files are stored persistently in agent state.

Parameters:
- file_path (required): Path where the file should be created/overwritten
- content (required): The complete content to write to the file

Important: This replaces the entire file content."""


# =============================================================================
# Tool Implementations
# =============================================================================


"""
[에이전트 상태에 저장된 가상 파일시스템의 모든 파일 목록 조회]

다른 파일 작업 수행 전 현재 에이전트 메모리에 존재하는 파일 확인 용도. 
파일 구성 현황 파악에 활용.

파라미터 불필요 - ls() 호출만으로 모든 사용 가능한 파일 조회 가능.
"""


@tool(description=LS_DESCRIPTION)
def ls(runtime: ToolRuntime) -> list[str]:
    """List all files in the virtual filesystem.

    Args:
        runtime: Runtime context with agent state (injected)

    Returns:
        List of file paths in the virtual filesystem
    """
    return list(runtime.state.get("files", {}).keys())


"""
[가상 파일시스템에서 파일 내용 읽기 (선택적 페이지네이션 지원)]

`cat -n`과 같이 라인 번호가 포함된 파일 내용 반환. 컨텍스트 오버플로우 방지를 위해 대용량 파일 청크 단위 읽기 지원.

파라미터:

- file_path (필수): 읽을 파일 경로
- offset (선택, 기본값=0): 읽기 시작 라인 번호
- limit (선택, 기본값=2000): 읽을 최대 라인 수


수정 전 기존 내용 파악 필수. 
편집 전 반드시 파일 읽기 선행.

1. 파일 전체를 읽어 오는 것이 아닌, 일부를 읽어오는 방식을 택함
2. 순차적으로 `offset` 부터 `limit` 만큼 읽어오는 방식을 택함
"""


@tool(description=READ_FILE_DESCRIPTION, parse_docstring=True)
def read_file(
    file_path: str,
    runtime: ToolRuntime,
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read file content from virtual filesystem with optional offset and limit.

    Args:
        file_path: Path to the file to read
        runtime: Runtime context with agent state (injected)
        offset: Line number to start reading from (default: 0)
        limit: Maximum number of lines to read (default: 2000)

    Returns:
        Formatted file content with line numbers, or error message if file not found
    """
    files = runtime.state.get("files", {})
    if file_path not in files:
        return f"Error: File '{file_path}' not found"

    content = files[file_path]
    if not content:
        return "System reminder: File exists but has empty contents"

    lines = content.splitlines()
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))

    if start_idx >= len(lines):
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

    result_lines = []
    for i in range(start_idx, end_idx):
        line_content = lines[i][:2000]  # Truncate long lines
        result_lines.append(f"{i + 1:6d}\t{line_content}")

    return "\n".join(result_lines)


"""
[가상 파일시스템에 새 파일 생성 또는 기존 파일 전체 덮어쓰기]

새 파일 생성 또는 파일 내용 전체 교체 용도. 초기 파일 생성 또는 전체 재작성 시 사용. 파일은 에이전트 상태에 영구 저장.

파라미터:

- file_path (필수): 파일 생성/덮어쓰기 경로
- content (필수): 파일에 작성할 전체 내용

주의: 파일 내용 전체 교체됨.
"""


@tool(description=WRITE_FILE_DESCRIPTION, parse_docstring=True)
def write_file(
    file_path: str,
    content: str,
    runtime: ToolRuntime,
) -> Command:
    """Write content to a file in the virtual filesystem.

    Args:
        file_path: Path where the file should be created/updated
        content: Content to write to the file
        runtime: Runtime context with agent state (injected)

    Returns:
        Command to update agent state with new file content
    """
    files = runtime.state.get("files", {})
    files[file_path] = content

    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(
                    f"Updated file {file_path}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
