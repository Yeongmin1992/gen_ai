"""DeepAgent 학습 패키지.

이 패키지는 DeepAgents 프레임워크의 핵심 패턴을 학습하기 위한 공통 모듈들을 제공합니다.

핵심 패턴:
1. TODO Planning - 작업 리스트 기반 플래닝
2. Context Offloading - 가상 파일 시스템을 통한 컨텍스트 관리
3. Sub-Agent Delegation - 컨텍스트 격리를 통한 작업 위임
"""

from .prompts import (
    FILE_USAGE_INSTRUCTIONS,
    SIMPLE_RESEARCH_INSTRUCTIONS,
    SUBAGENT_USAGE_INSTRUCTIONS,
    TODO_USAGE_INSTRUCTIONS,
)
from .state import DeepAgentState, TodoContent, file_reducer

__all__ = [
    # State
    "DeepAgentState",
    "TodoContent",
    "file_reducer",
    # Prompts
    "TODO_USAGE_INSTRUCTIONS",
    "FILE_USAGE_INSTRUCTIONS",
    "SUBAGENT_USAGE_INSTRUCTIONS",
    "SIMPLE_RESEARCH_INSTRUCTIONS",
]
