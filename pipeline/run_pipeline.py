from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .scoring import best_deviation_type, cue_summary, score_candidate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MAP = ROOT / "data" / "source_map.json"
DEFAULT_CANDIDATES = ROOT / "data" / "candidates.csv"
DEFAULT_OUT = ROOT / "site" / "data" / "dataset.json"
DEFAULT_REPORT = ROOT / "build" / "report.md"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_candidates(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def source_records(source_map: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for source in source_map["source_families"]:
        records.append(
            {
                "id": "source_" + slug(source["name"]),
                "record_type": "source",
                "title": source["name"],
                "url": source["url"],
                "domain": source["domain"],
                "source_family": source["name"],
                "priority": source["priority"],
                "deviation_type": source["deviation_type"],
                "score": max(1, 10 - int(source["priority"])),
                "status": "source_pointer",
                "summary": source["why_it_fits"],
                "best_use": source["best_use"],
                "annotation": None
            }
        )
    return records


def slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def intervention_window(timestamp: str) -> dict[str, int] | None:
    if not timestamp:
        return None
    try:
        center = max(0, int(float(timestamp)))
    except ValueError:
        return None
    return {
        "start_seconds": max(0, center - 30),
        "deviation_onset_seconds": center,
        "end_seconds": center + 30
    }


def assistant_draft(deviation_type: str, goal: str) -> str:
    if deviation_type == "route_failure":
        return "You may be moving away from the intended route. Pause and check the next turn for: " + goal
    if deviation_type == "search_failure":
        return "You may be near the target but not seeing it yet. Slow down and rescan the area for: " + goal
    if deviation_type == "omission":
        return "It looks like a required step or item may have been skipped. Check the plan before continuing: " + goal
    if deviation_type == "wrong_action":
        return "This looks like it may not match the intended plan. Confirm before proceeding with: " + goal
    if deviation_type == "correction":
        return "A correction may be needed now. Return to the last confirmed step for: " + goal
    return "Review this moment for a possible plan deviation related to: " + goal


def candidate_records(candidates: list[dict[str, str]], source_map: dict[str, Any], min_score: int) -> list[dict[str, Any]]:
    cues = source_map["deviation_cues"]
    records = []

    for row in candidates:
        score, hits = score_candidate(row, cues)
        deviation_type = best_deviation_type(hits)
        goal = row.get("goal", "").strip() or "Unknown goal"
        status = "candidate" if score >= min_score else "low_signal"
        window = intervention_window(row.get("timestamp_seconds", ""))

        records.append(
            {
                "id": row.get("id") or "candidate_" + str(len(records) + 1),
                "record_type": "candidate",
                "title": row.get("title", "").strip(),
                "url": row.get("url", "").strip(),
                "domain": row.get("domain", "").strip() or "unknown",
                "source_family": row.get("source_family", "").strip() or "unknown",
                "priority": 2,
                "deviation_type": deviation_type,
                "score": score,
                "status": status,
                "summary": row.get("description", "").strip(),
                "best_use": "Human or VLM verification of a visible plan-deviation moment.",
                "cue_hits": cue_summary(hits),
                "annotation": {
                    "goal": goal,
                    "current_state": "Needs visual review",
                    "expected_next_state": "Needs visual review",
                    "deviation_type": deviation_type,
                    "intervention_needed": score >= min_score,
                    "intervention_timing": "around transcript cue" if window else "needs localization",
                    "intervention_window": window,
                    "assistant_response": assistant_draft(deviation_type, goal),
                    "license": row.get("license", "review_required")
                }
            }
        )

    return sorted(records, key=lambda item: item["score"], reverse=True)


def write_outputs(manifest: dict[str, Any], out_path: Path, report_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    candidates = [record for record in manifest["records"] if record["record_type"] == "candidate"]
    sources = [record for record in manifest["records"] if record["record_type"] == "source"]
    valid_candidates = [record for record in candidates if record["status"] == "candidate"]

    lines = [
        "# Pipeline Report",
        "",
        f"Generated: {manifest['generated_at']}",
        f"Sources: {len(sources)}",
        f"Candidates: {len(candidates)}",
        f"Candidates above threshold: {len(valid_candidates)}",
        "",
        "## Top Candidates",
        ""
    ]
    for record in candidates[:10]:
        lines.append(f"- {record['id']}: score {record['score']} / {record['deviation_type']} / {record['title']}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(source_map_path: Path, candidates_path: Path, min_score: int) -> dict[str, Any]:
    source_map = read_json(source_map_path)
    candidates = read_candidates(candidates_path)
    records = source_records(source_map) + candidate_records(candidates, source_map, min_score)

    return {
        "name": source_map["benchmark_name"],
        "task": source_map["task"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "min_score": min_score,
        "deviation_cues": source_map["deviation_cues"],
        "records": records
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the trajectory-deviation dataset manifest.")
    parser.add_argument("--source-map", type=Path, default=DEFAULT_SOURCE_MAP)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--min-score", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.source_map, args.candidates, args.min_score)
    write_outputs(manifest, args.out, args.report)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.report}")
    print(f"Records: {len(manifest['records'])}")


if __name__ == "__main__":
    main()
