"""节点共享状态和模型结构化输出。

本模块定义了两类对象：

1) 结构化模型输出（`Plan` / `Research` / `Verdict`）
   - 用于约束各节点（Router / Researcher / Reviewer）返回的数据格式。
   - 通过 Pydantic 校验字段类型和长度约束，降低 LLM 非结构化输出导致的运行时错误。

2) 图执行共享状态（`GraphState`）
   - 用于在多节点执行流程中传递中间结果与最终结果。
   - 采用 `TypedDict(total=False)`，表示键是可选的、可按流程逐步填充。
"""
from qiye_rag import setting
from typing import Literal, TypedDict
from pydantic import BaseModel, Field


class Plan(BaseModel):
    """Router 节点的结构化输出。

    字段说明：
    - `kind`：问题路由类别，只允许固定枚举值。
      - `"chat"`：闲聊/无需检索。
      - `"simple"`：简单问题，通常单步检索可回答。
      - `"complex"`：复杂问题，通常需要多条子查询或多轮推理。
    - `queries`：Router 生成的独立检索子问题列表。
      - 默认空列表（避免使用可变默认值陷阱）。
      - 数量上限由 `setting.MAX_SUBQUERIES` 控制，防止子问题爆炸。
    """

    kind: Literal["chat", "simple", "complex"]
    queries: list[str] = Field(default_factory=list, max_length=setting.MAX_SUBQUERIES)


class Research(BaseModel):
    """Researcher 节点的结构化输出。

    字段说明：
    - `sufficient`：当前证据是否足够支撑回答。
      - `True`：可进入审阅或输出阶段。
      - `False`：通常需要继续补充检索。
    - `answer`：研究员给出的草稿回答文本。
      - 最大长度受 `setting.MAX_ANSWER_CHARS` 限制，控制上下文与成本。
    - `queries`：若证据不足，建议补充检索的问题列表。
      - 数量上限由 `setting.MAX_SUBQUERIES` 控制。
    """

    sufficient: bool
    answer: str = Field(max_length=setting.MAX_ANSWER_CHARS)
    queries: list[str] = Field(default_factory=list, max_length=setting.MAX_SUBQUERIES)


class Verdict(BaseModel):
    """Reviewer 节点的结构化输出。

    字段说明：
    - `grounded`：草稿是否被证据充分支持（事实可落地）。
    - `reason`：审阅结论理由（用于调试、回溯和重试决策）。
      - 最大长度受 `setting.MAX_REVIEW_REASON_CHARS` 限制。
    - `queries`：若不通过，可建议的补充检索问题。
      - 数量上限由 `setting.MAX_SUBQUERIES` 控制。
    """

    grounded: bool
    reason: str = Field(max_length=setting.MAX_REVIEW_REASON_CHARS)
    queries: list[str] = Field(default_factory=list, max_length=setting.MAX_SUBQUERIES)


class GraphState(TypedDict, total=False):
    """图执行过程中的共享状态容器（按需增量填充）。

    设计说明：
    - `total=False` 表示所有键均可选，方便节点按阶段写入。
    - `history` 通常跨轮保留，其余字段多为“当前轮次”中间结果。
    - 该类型只提供静态类型约束，不执行运行时校验（与 Pydantic 模型不同）。

    字段语义：
    - `question`：当前用户问题。
    - `history`：历史对话消息列表（如 `{role, content}` 结构）。
    - `kind`：Router 判定的问题类型（应对应 `Plan.kind`）。
    - `queries`：当前待检索/补充检索的问题列表。
    - `evidence`：检索到的证据片段集合（结构由上游检索器定义）。
    - `attempts`：当前轮重试/迭代次数。
    - `sufficient`：研究阶段判断证据是否充足（对应 `Research.sufficient`）。
    - `draft`：研究阶段生成的草稿回答。
    - `approved`：审阅是否通过（通常对应 `Verdict.grounded` 映射）。
    - `reason`：审阅结论或拒绝原因。
    - `answer`：最终对用户输出的答案文本。
    - `citations`：最终答案关联的引用来源列表。
    """

    question: str
    history: list[dict[str, str]]
    kind: str
    queries: list[str]
    evidence: list[dict]
    attempts: int
    sufficient: bool
    draft: str
    approved: bool
    reason: str
    answer: str
    citations: list[dict]
