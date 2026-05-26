from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class CueHit:
    category: str
    phrase: str
    field: str
    weight: int


FIELD_WEIGHTS = {
    "title": 3,
    "description": 2,
    "transcript": 1
}


def normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").lower()).strip()


def count_phrase(text: str, phrase: str) -> int:
    pattern = r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)"
    return len(re.findall(pattern, text))


def score_candidate(fields: Mapping[str, str], cues: Mapping[str, list[str]]) -> tuple[int, list[CueHit]]:
    hits: list[CueHit] = []
    score = 0

    for field, weight in FIELD_WEIGHTS.items():
        text = normalize_text(fields.get(field, ""))
        if not text:
            continue
        for category, phrases in cues.items():
            for phrase in phrases:
                occurrences = count_phrase(text, phrase)
                if occurrences:
                    score += occurrences * weight
                    hits.extend(CueHit(category, phrase, field, weight) for _ in range(occurrences))

    return score, hits


def best_deviation_type(hits: list[CueHit], fallback: str = "needs_review") -> str:
    counts: dict[str, int] = {}
    for hit in hits:
        counts[hit.category] = counts.get(hit.category, 0) + hit.weight
    if not counts:
        return fallback
    return max(counts.items(), key=lambda item: item[1])[0]


def cue_summary(hits: list[CueHit]) -> list[dict[str, str | int]]:
    return [
        {
            "category": hit.category,
            "phrase": hit.phrase,
            "field": hit.field,
            "weight": hit.weight
        }
        for hit in hits
    ]
