"""Small model outputs; source locations and quotations are verified by code."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Text = Annotated[str, Field(min_length=1, max_length=4000)]
Dimension = Literal["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]
Strength = Literal["明确可见", "有较强迹象", "证据不足"]
Utilization = Literal["充分利用", "部分利用", "未利用", "误用", "无法判断"]
Effect = Literal["正向", "无明显变化", "负向", "待验证"]
DIMENSIONS = {
    "A1": "长程信息复用",
    "A2": "减少无效重复搜索",
    "A3": "定位关键证据缺口",
    "A4": "合理分配验证精力",
    "A5": "发现冲突并修正判断",
    "A6": "结论与证据可追溯",
    "A7": "降低整体上下文开销",
    "A8": "改善多 Agent 协作",
}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Finding(Strict):
    anchor_id: str
    kind: Literal["benefit", "missed_opportunity", "misuse"]
    label: Annotated[str, Field(min_length=1, max_length=120)]
    dimensions: Annotated[list[Dimension], Field(min_length=1, max_length=8)]
    graph_quotes: Annotated[list[Text], Field(min_length=1, max_length=4)]
    action_quote: Text
    report_quote: str = Field(default="", max_length=4000)
    related_ids: list[str] = Field(default_factory=list, max_length=8)
    expected: Text
    observed: Text
    impact: Text
    alternative_explanation: Text
    strength: Strength


class Findings(Strict):
    findings: list[Finding] = Field(max_length=8)
    limitations: list[Text] = Field(max_length=12)


class Rating(Strict):
    dimension: Dimension
    finding_ids: Annotated[list[str], Field(min_length=1, max_length=12)]
    utilization: Utilization
    effect: Effect
    note: Text


class Conclusion(Strict):
    benefit: str = Field(max_length=5000)
    problem: str = Field(max_length=5000)
    evidence: Strength
    improvement: Text
    comparison: Text
    verification: Text


class Summary(Strict):
    goal: Text
    selected_finding_ids: list[str] = Field(max_length=12)
    ratings: list[Rating] = Field(max_length=8)
    conclusion: Conclusion
