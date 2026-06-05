"""Unit tests for the JSON parsing logic — no GPU needed."""

import pytest

from core.inference import _parse_result

_FULL_THINKING_OUTPUT = """{
  "thinking": {
    "observations": ["The pothole area shows fresh dark asphalt fill.", "Surface appears mostly flush with the surrounding road."],
    "positive_evidence": ["Dark asphalt material fills the void completely.", "No visible depression relative to road level."],
    "negative_evidence": ["A hairline crack runs along the left edge of the patch.", "Loose gravel present in a 5 cm strip near the kerb."],
    "reasoning": "The main body of the repair meets the required state. However, the edge seal shows early cracking and loose aggregate — both are listed failure signals. The crack is minor but present, reducing confidence below the threshold."
  },
  "passed": false,
  "confidence": 0.61,
  "rationale": "Repair incomplete — edge seal shows cracking and loose gravel at left margin."
}"""


def test_full_thinking_output_parsed():
    result = _parse_result(_FULL_THINKING_OUTPUT)
    assert result.passed is False
    assert result.confidence == pytest.approx(0.61)
    assert result.thinking is not None
    assert len(result.thinking.observations) == 2
    assert len(result.thinking.positive_evidence) == 2
    assert len(result.thinking.negative_evidence) == 2
    assert "edge seal" in result.thinking.reasoning


def test_thinking_positive_pass():
    raw = """{
      "thinking": {
        "observations": ["Smooth silver bumper surface, uniform paint sheen."],
        "positive_evidence": ["No visible dent or crease on the bumper panel."],
        "negative_evidence": ["none observed"],
        "reasoning": "All visual indicators confirm the repair is complete."
      },
      "passed": true,
      "confidence": 0.94,
      "rationale": "Dent fully repaired, surface matches surrounding body panels."
    }"""
    result = _parse_result(raw)
    assert result.passed is True
    assert result.confidence == pytest.approx(0.94)
    assert result.thinking is not None
    assert result.thinking.negative_evidence == ["none observed"]


def test_native_think_tags_stripped_before_parse():
    raw = (
        "<think>Let me carefully inspect the image...</think>"
        '{"thinking": {"observations": ["Clean grout lines."], "positive_evidence": ["No mold."], '
        '"negative_evidence": ["none observed"], "reasoning": "Surface is clean."}, '
        '"passed": true, "confidence": 0.91, "rationale": "No mold detected."}'
    )
    result = _parse_result(raw)
    assert result.passed is True
    assert result.confidence == pytest.approx(0.91)
    assert result.thinking is not None


def test_missing_thinking_block_still_parses():
    raw = '{"passed": true, "confidence": 0.94, "rationale": "Surface is flush."}'
    result = _parse_result(raw)
    assert result.passed is True
    assert result.confidence == pytest.approx(0.94)
    assert result.thinking is None


def test_json_embedded_in_prose():
    raw = 'Sure! Here is my evaluation: {"passed": false, "confidence": 0.72, "rationale": "Loose gravel visible."} Hope that helps!'
    result = _parse_result(raw)
    assert result.passed is False
    assert result.confidence == pytest.approx(0.72)


def test_completely_malformed_output_returns_safe_default():
    raw = "I cannot evaluate this image."
    result = _parse_result(raw)
    assert result.passed is False
    assert result.confidence == 0.0
    assert "Parse failure" in result.rationale


def test_missing_keys_returns_safe_default():
    raw = '{"passed": true}'  # missing confidence and rationale
    result = _parse_result(raw)
    assert result.passed is False
    assert result.confidence == 0.0


def test_confidence_boundary_values():
    raw = '{"passed": true, "confidence": 1.0, "rationale": "Perfect."}'
    result = _parse_result(raw)
    assert result.confidence == pytest.approx(1.0)

    raw = '{"passed": false, "confidence": 0.0, "rationale": "Total failure."}'
    result = _parse_result(raw)
    assert result.confidence == pytest.approx(0.0)
