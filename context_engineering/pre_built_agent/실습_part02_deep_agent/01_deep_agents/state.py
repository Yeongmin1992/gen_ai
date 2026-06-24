"""DeepAgent 상태(State) 정의 모듈.

- TodoContent: 작업 진행 상황 추적용 구조
- file_reducer: 가상 파일 시스템 병합 함수
- DeepAgentState: TODO 리스트와 가상 파일 시스템을 포함하는 확장 상태
"""

from typing import Annotated, Literal, NotRequired, TypedDict

from langchain.agents import AgentState


class TodoContent(TypedDict):
    """복잡한 작업 플로우의 진행 상황을 추적하기 위한 구조화된 작업 항목.

    Attributes:
        content: 작업에 대한 짧고 구체적인 설명
        status: 현재 상태 - pending(대기), in_progress(진행 중), completed(완료)
    """

    content: str
    status: Literal["pending", "in_progress", "completed"]


def file_reducer(
    left: dict[str, str] | None, right: dict[str, str] | None
) -> dict[str, str]:
    """두 파일 딕셔너리를 병합하며, 오른쪽 값이 우선 적용됨.

    에이전트 상태의 files 필드에 대한 reducer 함수로 사용되며,
    가상 파일 시스템에 대한 점진적 업데이트를 가능하게 함.

    Args:
        left: 기존 파일들 (왼쪽 딕셔너리)
        right: 새로운/업데이트된 파일들 (오른쪽 딕셔너리)

    Returns:
        오른쪽 값이 왼쪽 값을 덮어쓴 병합된 딕셔너리
    """
    if left is None:
        return right or {}
    elif right is None:
        return left
    else:
        return {**left, **right}


class DeepAgentState(AgentState):
    """작업 추적 및 가상 파일 시스템을 포함하는 확장된 에이전트 상태.

    LangGraph의 AgentState를 상속하며 다음을 추가:
    - todos: 작업 계획 및 진행 상황 추적을 위한 Todo 항목 리스트
    - files: 파일명을 내용에 매핑하는 딕셔너리 형태의 가상 파일 시스템

    Notes:
        - todos는 전체 덮어쓰기 방식으로 업데이트
        - files는 file_reducer로 병합 (새로운/업데이트된 파일들 (오른쪽 딕셔너리 우선))
    """

    todos: NotRequired[list[TodoContent]]
    files: Annotated[NotRequired[dict[str, str]], file_reducer]
