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
DEFAULT_VERIFICATIONS = ROOT / "data" / "verifications.json"
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


def read_verifications(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"records": {}}
    return read_json(path)


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


def curated_window(row: dict[str, str]) -> dict[str, int] | None:
    start = row.get("start_seconds", "").strip()
    onset = row.get("deviation_onset_seconds", "").strip() or row.get("timestamp_seconds", "").strip()
    end = row.get("end_seconds", "").strip()
    if start and onset and end:
        try:
            return {
                "start_seconds": max(0, int(float(start))),
                "deviation_onset_seconds": max(0, int(float(onset))),
                "end_seconds": max(0, int(float(end)))
            }
        except ValueError:
            return None
    return intervention_window(onset)


def row_value(row: dict[str, str], key: str, fallback: str = "") -> str:
    value = row.get(key, "")
    return value.strip() if value else fallback


def row_deviation_type(row: dict[str, str], hits: list[Any]) -> str:
    return row_value(row, "deviation_type") or best_deviation_type(hits)


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


def verification_for(row: dict[str, str], verifications: dict[str, Any]) -> dict[str, Any] | None:
    record_id = row_value(row, "id")
    wrapper = verifications.get("records", {}).get(record_id)
    if not wrapper:
        return None
    return wrapper.get("result")


def verified_int(value: Any, fallback: int) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return fallback


def verified_window(result: dict[str, Any] | None, fallback: dict[str, int] | None) -> dict[str, int] | None:
    if not result:
        return fallback
    return {
        "start_seconds": verified_int(result.get("start_seconds"), fallback["start_seconds"] if fallback else 0),
        "deviation_onset_seconds": verified_int(
            result.get("deviation_onset_seconds"),
            fallback["deviation_onset_seconds"] if fallback else 0
        ),
        "end_seconds": verified_int(result.get("end_seconds"), fallback["end_seconds"] if fallback else 0)
    }


def candidate_records(
    candidates: list[dict[str, str]],
    source_map: dict[str, Any],
    verifications: dict[str, Any],
    min_score: int
) -> list[dict[str, Any]]:
    cues = source_map["deviation_cues"]
    records = []

    for row in candidates:
        score, hits = score_candidate(row, cues)
        verification = verification_for(row, verifications)
        fallback_window = curated_window(row)
        window = verified_window(verification, fallback_window)
        verifier_valid = bool(verification.get("valid")) if verification else False
        verifier_available = verification is not None
        deviation_type = (
            verification.get("deviation_type")
            if verification and verification.get("deviation_type")
            else row_deviation_type(row, hits)
        )
        goal = (
            verification.get("goal")
            if verification and verification.get("goal")
            else row_value(row, "goal", "Unknown goal")
        )
        if verifier_available:
            status = "curated" if verifier_valid else "rejected"
        else:
            status = "awaiting_vlm"
        assistant_response = (
            verification.get("assistant_response")
            if verification and verification.get("assistant_response")
            else assistant_draft(deviation_type, goal)
        )

        records.append(
            {
                "id": row_value(row, "id") or "candidate_" + str(len(records) + 1),
                "record_type": "candidate",
                "title": row_value(row, "title"),
                "url": row_value(row, "url"),
                "domain": row_value(row, "domain", "unknown"),
                "source_family": row_value(row, "source_family", "unknown"),
                "priority": 2,
                "deviation_type": deviation_type,
                "score": score,
                "status": status,
                "summary": row_value(row, "description"),
                "best_use": "Human or VLM verification of a visible plan-deviation moment.",
                "cue_hits": cue_summary(hits),
                "verifier": verification,
                "curation": {
                    "status": status,
                    "reviewer": "gemini" if verifier_available else "",
                    "visual_evidence": (
                        verification.get("visual_evidence")
                        if verification and verification.get("visual_evidence")
                        else row_value(row, "visual_evidence")
                    ),
                    "notes": (
                        verification.get("rejection_reason")
                        if verification and verification.get("rejection_reason")
                        else row_value(row, "notes")
                    )
                },
                "rights": {
                    "license": row_value(row, "license", "review_required"),
                    "license_status": row_value(row, "license_status", "unknown"),
                    "hosting_status": row_value(row, "hosting_status", "external_link_only")
                },
                "annotation": {
                    "goal": goal,
                    "current_state": (
                        verification.get("current_state")
                        if verification and verification.get("current_state")
                        else row_value(row, "current_state", "Needs VLM verification")
                    ),
                    "expected_next_state": (
                        verification.get("expected_next_state")
                        if verification and verification.get("expected_next_state")
                        else row_value(row, "expected_next_state", "Needs VLM verification")
                    ),
                    "deviation_type": deviation_type,
                    "intervention_needed": bool(verification.get("intervention_needed")) if verification else False,
                    "intervention_timing": (
                        verification.get("intervention_timing")
                        if verification and verification.get("intervention_timing")
                        else ("around transcript cue" if window else "needs localization")
                    ),
                    "intervention_window": window,
                    "assistant_response": assistant_response
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
    valid_candidates = [record for record in candidates if record["status"] == "curated"]
    curated_candidates = [record for record in candidates if record["status"] == "curated"]
    awaiting_vlm = [record for record in candidates if record["status"] == "awaiting_vlm"]
    rejected = [record for record in candidates if record["status"] == "rejected"]

    lines = [
        "# Pipeline Report",
        "",
        f"Generated: {manifest['generated_at']}",
        f"Sources: {len(sources)}",
        f"Candidates: {len(candidates)}",
        f"VLM-valid candidates: {len(valid_candidates)}",
        f"Curated candidates: {len(curated_candidates)}",
        f"Awaiting VLM: {len(awaiting_vlm)}",
        f"Rejected by VLM: {len(rejected)}",
        "",
        "## Top Candidates",
        ""
    ]
    for record in candidates[:10]:
        lines.append(
            f"- {record['id']}: score {record['score']} / {record['status']} / "
            f"{record['deviation_type']} / {record['title']}"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(
    source_map_path: Path,
    candidates_path: Path,
    verifications_path: Path,
    min_score: int,
    verified_only: bool
) -> dict[str, Any]:
    source_map = read_json(source_map_path)
    candidates = read_candidates(candidates_path)
    verifications = read_verifications(verifications_path)
    candidates_out = candidate_records(candidates, source_map, verifications, min_score)
    if verified_only:
        records = [record for record in candidates_out if record["status"] == "curated"]
    else:
        records = source_records(source_map) + candidates_out

    return {
        "name": source_map["benchmark_name"],
        "task": source_map["task"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "min_score": min_score,
        "verified_only": verified_only,
        "deviation_cues": source_map["deviation_cues"],
        "records": records
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the trajectory-deviation dataset manifest.")
    parser.add_argument("--source-map", type=Path, default=DEFAULT_SOURCE_MAP)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--verifications", type=Path, default=DEFAULT_VERIFICATIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--min-score", type=int, default=3)
    parser.add_argument("--verified-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(
        args.source_map,
        args.candidates,
        args.verifications,
        args.min_score,
        args.verified_only
    )
    write_outputs(manifest, args.out, args.report)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.report}")
    print(f"Records: {len(manifest['records'])}")


if __name__ == "__main__":
    main()
