"""Unit tests for the JSON parsing logic — no GPU needed."""

import pytest

from core.inference import _parse_result


def test_clean_json_output():
    raw = '{"passed": true, "confidence": 0.94, "rationale": "Surface is flush."}'
    result = _parse_result(raw)
    assert result.passed is True
    assert result.confidence == pytest.approx(0.94)
    assert "flush" in result.rationale


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
