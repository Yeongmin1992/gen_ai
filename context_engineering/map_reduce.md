# Map-Reduce 패턴

일반 개념부터 LangGraph / 비동기 에이전트 맥락까지 정리한 문서입니다.

## 1. 큰 그림: Map-Reduce란

이름 그대로 두 단계로 나뉩니다.

- **Map (나누고 → 각각 처리)**: 큰 작업을 여러 개의 독립적인 조각으로 쪼개고, 각 조각을 **동시에/병렬로** 처리합니다.
- **Reduce (모으고 → 합치기)**: 흩어진 결과들을 하나로 **취합·요약·집계**합니다.

```
                 ┌─→ [Map] 조각1 처리 ─┐
입력 → [분할] ───┼─→ [Map] 조각2 처리 ─┼─→ [Reduce] 결과 합치기 → 출력
                 └─→ [Map] 조각3 처리 ─┘
                  (이 3개가 동시에 실행됨)
```

핵심 전제는 **"각 Map 작업이 서로 독립적"** 이라는 것입니다. 조각2가 조각1의 결과에 의존하면 병렬화할 수 없습니다.

## 2. 왜 쓰는가 (특히 LLM 에이전트에서)

LLM 호출은 **느린 I/O 작업**입니다 (한 번에 수 초). 그래서 직렬로 처리하면 시간이 선형으로 늘어납니다.

```python
# ❌ 직렬: 문서 10개 요약 → 3초 × 10 = 30초
for doc in docs:
    summary = await summarize(doc)

# ✅ Map-Reduce: 10개 동시 요약 → 약 3초
summaries = await asyncio.gather(*[summarize(doc) for doc in docs])
final = await combine(summaries)   # Reduce
```

LLM 호출은 대부분 "응답을 기다리는" 시간이라, 논블로킹 `await` 덕분에 일꾼(이벤트 루프)이 대기 시간에 다른 호출을 동시에 진행시킵니다. 그래서 Map 단계가 극적으로 빨라집니다.

## 3. 대표적인 활용 사례

| 사례 | Map | Reduce |
|---|---|---|
| 긴 문서 요약 | 청크별 부분 요약 | 부분 요약들을 종합 |
| 다중 문서 RAG | 문서별 관련성 평가/추출 | 근거 종합해 답변 생성 |
| 멀티 에이전트 리서치 | 하위 질문별 병렬 조사 | 리서치 결과 통합 리포트 |
| 평가/채점 | 항목별 독립 채점 | 점수 집계 |

## 4. 순수 asyncio 구현

```python
import asyncio

async def map_step(item):
    # 각 조각을 독립적으로 처리 (예: LLM 호출)
    await asyncio.sleep(1)
    return f"{item} 처리완료"

async def reduce_step(results):
    # 흩어진 결과를 하나로 합침
    return " / ".join(results)

async def map_reduce(items):
    # Map: 전부 동시에 출발시키고 모든 결과를 기다림
    mapped = await asyncio.gather(*[map_step(x) for x in items])
    # Reduce: 합치기
    return await reduce_step(mapped)

result = asyncio.run(map_reduce(["A", "B", "C"]))
```

여기서 `asyncio.gather`가 Map 병렬 실행의 핵심입니다. 셋이 각각 1초씩이지만 **겹쳐서** 돌기 때문에 전체가 약 1초입니다.

## 5. LangGraph에서의 Map-Reduce — `Send` API

LangGraph에서는 Map-Reduce를 **`Send`** 라는 메커니즘으로 구현합니다.

문제 상황: 보통 그래프의 엣지는 **정적**입니다(노드 A → 노드 B 고정). 그런데 Map-Reduce는 **런타임에 조각 개수가 정해집니다** (문서가 3개일지 100개일지 미리 모름). 이 "동적 팬아웃(fan-out)"을 `Send`가 해결합니다.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing import Annotated
import operator
from typing_extensions import TypedDict

class State(TypedDict):
    topic: str
    subtasks: list[str]
    # Reduce의 핵심: 여러 노드가 동시에 쓰는 값은 reducer로 합쳐야 함
    results: Annotated[list, operator.add]

# 1) 분할 노드: 작업을 조각으로 나눔
def split(state: State):
    return {"subtasks": ["조각1", "조각2", "조각3"]}

# 2) 라우팅 함수: 각 조각마다 worker 노드로 Send (← 동적 팬아웃 = Map)
def assign_workers(state: State):
    return [Send("worker", {"task": t}) for t in state["subtasks"]]

# 3) Worker 노드: 조각 하나를 처리
def worker(state: dict):
    result = f"{state['task']} 결과"   # 실제로는 LLM 호출
    return {"results": [result]}       # operator.add로 자동 누적

# 4) Reduce 노드: 모인 결과를 합침
def reduce_node(state: State):
    return {"final": "\n".join(state["results"])}

builder = StateGraph(State)
builder.add_node("split", split)
builder.add_node("worker", worker)
builder.add_node("reduce_node", reduce_node)

builder.add_edge(START, "split")
builder.add_conditional_edges("split", assign_workers, ["worker"])  # Map 팬아웃
builder.add_edge("worker", "reduce_node")  # 모든 worker가 끝나면 reduce로
builder.add_edge("reduce_node", END)
```

### 여기서 꼭 이해할 두 가지

**(1) `Send("worker", {...})`**
"worker 노드를 이 입력으로 실행해줘"를 **리스트로 여러 개** 반환합니다. 리스트 길이만큼 worker가 **병렬 인스턴스**로 생성됩니다. 이게 Map의 동적 팬아웃입니다. 각 worker는 전체 State가 아니라 `Send`로 넘긴 **자기 조각만** 받습니다.

**(2) `Annotated[list, operator.add]` — Reducer**
여러 worker가 **동시에** `results`에 쓰면 충돌이 납니다. LangGraph는 이때 "어떻게 합칠지"를 reducer로 지정하게 합니다. `operator.add`는 리스트를 이어붙여서, 흩어진 worker 결과가 자동으로 한 리스트에 모입니다. **이 자동 취합이 Reduce 단계의 절반**이고, `reduce_node`가 나머지(최종 요약/집계)를 담당합니다.

> reducer를 지정하지 않으면 마지막 worker의 값이 앞 값을 덮어써 버립니다 — 병렬 쓰기에서 가장 흔한 버그입니다.

## 6. 실무에서 주의할 점

1. **에러 처리**: Map 작업 하나가 실패하면? `asyncio.gather(..., return_exceptions=True)`로 한 조각 실패가 전체를 무너뜨리지 않게 합니다.
2. **동시성 제한**: 1,000개를 한꺼번에 띄우면 API rate limit에 걸립니다. `asyncio.Semaphore`로 동시 실행 수를 제한하세요.
   ```python
   sem = asyncio.Semaphore(10)  # 최대 10개 동시
   async def limited(item):
       async with sem:
           return await map_step(item)
   ```
3. **순서 보장**: `gather`는 입력 순서대로 결과를 돌려줍니다. 단, `Send`로 모인 결과의 순서는 완료 순서에 따라 달라질 수 있으니 순서가 중요하면 인덱스를 함께 넘기세요.
4. **Reduce가 너무 크면**: 결과가 많을 때 한 번에 합치기 어려우면 **계층적 Reduce**(부분끼리 먼저 합치고 → 그걸 다시 합침)를 씁니다. 긴 문서 요약의 전형적 패턴입니다.

## 한 줄 요약

Map-Reduce = **독립적인 조각으로 쪼개 병렬 처리(Map)한 뒤 결과를 하나로 합치는(Reduce)** 패턴. LLM 에이전트에서는 느린 LLM 호출을 `asyncio.gather`(순수 비동기)나 LangGraph의 **`Send` + reducer**(`operator.add`)로 겹쳐 실행해 속도를 끌어올리는 데 씁니다.
