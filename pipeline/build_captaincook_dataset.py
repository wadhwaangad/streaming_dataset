from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERIFICATIONS = ROOT / "data" / "captaincook_qwen_verifications.json"
DEFAULT_OUT = ROOT / "site" / "data" / "captaincook_dataset.json"
DEFAULT_REPORT = ROOT / "build" / "captaincook_report.md"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"records": {}}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def source_url(record: dict[str, Any]) -> str:
    row = record.get("source_row", {})
    video = row.get("video", "")
    return video if isinstance(video, str) else ""


def build_record(record_id: str, wrapper: dict[str, Any]) -> dict[str, Any]:
    result = wrapper.get("result", {})
    row = wrapper.get("source_row", {})
    goal = result.get("goal") or row.get("recipe") or "CaptainCook4D cooking procedure"
    start = int(result.get("start_seconds") or wrapper.get("clip_start_seconds") or 0)
    onset = int(result.get("deviation_onset_seconds") or start)
    end = int(result.get("end_seconds") or wrapper.get("clip_end_seconds") or start)
    status = "curated" if result.get("valid") else "rejected"
    url = source_url(wrapper)
    return {
        "id": record_id,
        "source_candidate_id": wrapper.get("source_candidate_id", ""),
        "record_type": "captaincook_candidate",
        "title": goal,
        "url": url,
        "clip_url": url,
        "intervention_url": url,
        "deviation_onset_url": url,
        "domain": "cooking",
        "source_family": "CaptainCook4D",
        "priority": 1,
        "deviation_type": result.get("deviation_type", "not_a_deviation"),
        "score": int(float(result.get("confidence") or 0) * 10),
        "status": status,
        "summary": result.get("visual_evidence") or row.get("label") or "",
        "best_use": "Local Qwen2.5-VL verification of CaptainCook4D procedural deviations.",
        "verifier": result,
        "curation": {
            "status": status,
            "reviewer": "qwen2.5-vl-local",
            "visual_evidence": result.get("visual_evidence", ""),
            "notes": result.get("rejection_reason", ""),
        },
        "rights": {
            "license": "review_required",
            "license_status": "restricted",
            "hosting_status": "metadata_only",
        },
        "annotation": {
            "goal": goal,
            "current_state": result.get("current_state", ""),
            "expected_next_state": result.get("expected_next_state", ""),
            "deviation_type": result.get("deviation_type", "not_a_deviation"),
            "intervention_needed": bool(result.get("intervention_needed")),
            "intervention_timing": result.get("intervention_timing", ""),
            "intervention_window": {
                "start_seconds": start,
                "deviation_onset_seconds": onset,
                "end_seconds": end,
            },
            "assistant_response": result.get("assistant_response", ""),
        },
    }


def build_manifest(verifications_path: Path, verified_only: bool) -> dict[str, Any]:
    verifications = read_json(verifications_path)
    records = [
        build_record(record_id, wrapper)
        for record_id, wrapper in verifications.get("records", {}).items()
    ]
    if verified_only:
        records = [record for record in records if record["status"] == "curated"]
    return {
        "name": "CaptainCook4D Local Qwen Verification",
        "task": "Detect procedural cooking deviations from CaptainCook4D clips using a local Qwen2.5-VL 3B verifier.",
        "dataset": verifications.get("dataset", "SabrianLinnn/captain_cook_4d"),
        "model": verifications.get("model", "Qwen/Qwen2.5-VL-3B-Instruct"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verified_only": verified_only,
        "records": records,
    }


def write_report(manifest: dict[str, Any], path: Path) -> None:
    records = manifest["records"]
    curated = [record for record in records if record["status"] == "curated"]
    rejected = [record for record in records if record["status"] == "rejected"]
    lines = [
        "# CaptainCook4D Qwen Report",
        "",
        f"Generated: {manifest['generated_at']}",
        f"Dataset: {manifest['dataset']}",
        f"Model: {manifest['model']}",
        f"Records: {len(records)}",
        f"Curated: {len(curated)}",
        f"Rejected: {len(rejected)}",
        "",
        "## Recent Records",
        "",
    ]
    for record in records[:10]:
        lines.append(f"- {record['id']}: {record['status']} / {record['deviation_type']} / {record['title']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the CaptainCook4D-specific website manifest.")
    parser.add_argument("--verifications", type=Path, default=DEFAULT_VERIFICATIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--verified-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.verifications, args.verified_only)
    write_json(args.out, manifest)
    write_report(manifest, args.report)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.report}")
    print(f"Records: {len(manifest['records'])}")


if __name__ == "__main__":
    main()
