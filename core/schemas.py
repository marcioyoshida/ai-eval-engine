from pydantic import BaseModel, Field


class ContractParams(BaseModel):
    target_object: str
    required_state: str
    negative_indicators: list[str] = Field(default_factory=list)
    strictness_coefficient: float = Field(default=0.80, ge=0.0, le=1.0)


class EvaluationResult(BaseModel):
    passed: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    # set by the evaluation layer after applying strictness_coefficient
    threshold_applied: float | None = None
    routed_to_human: bool = False
