from pydantic import BaseModel, Field


class ContractParams(BaseModel):
    target_object: str
    required_state: str
    negative_indicators: list[str] = Field(default_factory=list)
    strictness_coefficient: float = Field(default=0.80, ge=0.0, le=1.0)


class EvaluationThinking(BaseModel):
    observations: list[str] = Field(
        description="What the model directly sees in the image, stated as neutral facts."
    )
    positive_evidence: list[str] = Field(
        description="Specific visual details that support a PASS verdict."
    )
    negative_evidence: list[str] = Field(
        description="Specific visual details that support a FAIL verdict."
    )
    reasoning: str = Field(
        description="Step-by-step chain of thought weighing evidence before reaching the verdict."
    )


class EvaluationResult(BaseModel):
    passed: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    thinking: EvaluationThinking | None = None
    # set by the evaluation layer after applying strictness_coefficient
    threshold_applied: float | None = None
    routed_to_human: bool = False
