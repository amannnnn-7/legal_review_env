from __future__ import annotations

import re
from typing import Iterable

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from .models import GradeReport, TaskDifficulty


_SCORE_EPSILON = 0.001


_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "eighteen": 18,
    "twenty": 20,
    "twenty-four": 24,
    "twenty four": 24,
}

_DURATION_PATTERN = re.compile(
    r"(?ix)\b(?P<value>\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|eighteen|twenty(?:[- ]four)?)\b(?:\s*\(\s*\d+\s*\))?\s*(?P<unit>day|days|month|months|year|years)\b"
)
_HYPHENATED_YEAR_PATTERN = re.compile(
    r"(?ix)\b(?P<value>\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|eighteen|twenty(?:[- ]four)?)-year\b"
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def parse_duration_months(text: str) -> float | None:
    lowered = text.lower().replace("‑", "-").replace("–", "-")
    if "perpetual" in lowered or "indefinite" in lowered:
        return 10_000.0

    values: list[float] = []
    for match in _DURATION_PATTERN.finditer(lowered):
        raw_value = match.group("value")
        if raw_value.replace(".", "", 1).isdigit():
            value = float(raw_value)
        else:
            value = float(_WORD_NUMBERS[raw_value])

        unit = match.group("unit")
        if unit.startswith("day"):
            values.append(value / 30.0)
        elif unit.startswith("year"):
            values.append(value * 12.0)
        else:
            values.append(value)

    return max(values) if values else None


def non_compete_is_violation(text: str) -> bool:
    months = parse_duration_months(text)
    return months is None or months > 12.0


def find_enclosing_block(context: str, answer_start: int, answer_text: str) -> str:
    boundaries = list(re.finditer(r"\n\s*\n", context))

    start = 0
    for boundary in boundaries:
        if boundary.start() < answer_start:
            start = boundary.end()
        else:
            break

    end_anchor = answer_start + len(answer_text)
    end = len(context)
    for boundary in boundaries:
        if boundary.start() > end_anchor:
            end = boundary.start()
            break

    return context[start:end].strip()


def rewrite_non_compete_block(block: str) -> str:
    def replace_duration(match: re.Match[str]) -> str:
        matched = match.group(0)
        months = parse_duration_months(matched)
        if months is None or months <= 12.0:
            return matched
        return "12 months"

    rewritten = _HYPHENATED_YEAR_PATTERN.sub(
        lambda m: "12-month" if parse_duration_months(m.group(0)) and parse_duration_months(m.group(0)) > 12.0 else m.group(0),
        block,
    )
    rewritten = _DURATION_PATTERN.sub(replace_duration, rewritten)
    rewritten = re.sub(r"(?i)\b(perpetual|indefinite)\b", "12 months", rewritten)
    return rewritten


def _greedy_match(predicted: Iterable[str], expected: Iterable[str]) -> tuple[int, int, int]:
    predicted_norm = [normalize_text(item) for item in predicted if item.strip()]
    expected_norm = [normalize_text(item) for item in expected if item.strip()]
    used_expected: set[int] = set()
    true_positives = 0

    for predicted_text in predicted_norm:
        for index, expected_text in enumerate(expected_norm):
            if index in used_expected:
                continue
            if (
                predicted_text == expected_text
                or predicted_text in expected_text
                or expected_text in predicted_text
            ):
                used_expected.add(index)
                true_positives += 1
                break

    false_positives = max(0, len(predicted_norm) - true_positives)
    false_negatives = max(0, len(expected_norm) - true_positives)
    return true_positives, false_positives, false_negatives


def _open_interval_score(raw_score: float) -> float:
    bounded = min(max(raw_score, 0.0), 1.0)
    return round(_SCORE_EPSILON + ((1.0 - (2.0 * _SCORE_EPSILON)) * bounded), 4)


def grade_easy(
    extracted: dict[str, list[str]],
    expected: dict[str, tuple[str, ...]],
) -> GradeReport:
    category_scores: dict[str, float] = {}
    for category, expected_values in expected.items():
        actual_values = tuple(extracted.get(category, []))
        category_scores[category] = 1.0 if tuple(actual_values) == expected_values else 0.0

    raw_score = sum(category_scores.values()) / max(len(category_scores), 1)
    score = _open_interval_score(raw_score)
    return GradeReport(
        difficulty=TaskDifficulty.EASY,
        metric="exact_match_average",
        score=score,
        complete=raw_score >= 1.0,
        details={
            "raw_score": round(raw_score, 4),
            "categories": category_scores,
            "expected": expected,
            "actual": extracted,
        },
    )


def grade_medium(flagged: Iterable[str], expected: Iterable[str]) -> GradeReport:
    true_positives, false_positives, false_negatives = _greedy_match(flagged, expected)
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    if precision + recall == 0.0:
        raw_score = 0.0
    else:
        raw_score = 2 * precision * recall / (precision + recall)
    score = _open_interval_score(raw_score)

    return GradeReport(
        difficulty=TaskDifficulty.MEDIUM,
        metric="span_f1",
        score=score,
        complete=raw_score >= 1.0,
        details={
            "raw_score": round(raw_score, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "expected": list(expected),
            "predicted": list(flagged),
        },
    )


def grade_hard(current_block: str, original_block: str, target_block: str) -> GradeReport:
    if normalize_text(current_block) == normalize_text(original_block):
        raw_score = 0.0
        similarity = 0.0
        edit_penalty = 0.0
    else:
        normalized_current = normalize_text(current_block)
        normalized_target = normalize_text(target_block)
        normalized_original = normalize_text(original_block)

        similarity = fuzz.ratio(normalized_current, normalized_target) / 100.0
        edit_penalty = Levenshtein.distance(normalized_original, normalized_current) / max(
            len(normalized_original), 1
        )
        raw_score = max(0.0, min(1.0, similarity - (0.35 * edit_penalty)))

    score = _open_interval_score(raw_score)

    return GradeReport(
        difficulty=TaskDifficulty.HARD,
        metric="similarity_minus_edit_penalty",
        score=score,
        complete=raw_score >= 0.999,
        details={
            "raw_score": round(raw_score, 4),
            "similarity_to_target": round(similarity, 4),
            "edit_penalty": round(edit_penalty, 4),
            "original": original_block,
            "current": current_block,
            "target": target_block,
        },
    )