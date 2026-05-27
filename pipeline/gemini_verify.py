from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = ROOT / "data" / "candidates.csv"
DEFAULT_OUT = ROOT / "data" / "verifications.json"
DEFAULT_MODEL = "gemini-2.5-flash"
API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "valid": {"type": "boolean"},
        "confidence": {"type": "number"},
        "goal": {"type": "string"},
        "current_state": {"type": "string"},
        "expected_next_state": {"type": "string"},
        "deviation_type": {"type": "string"},
        "visual_evidence": {"type": "string"},
        "could_assist_before_user_realizes": {"type": "boolean"},
        "intervention_needed": {"type": "boolean"},
        "intervention_timing": {"type": "string"},
        "start_seconds": {"type": "integer"},
        "deviation_onset_seconds": {"type": "integer"},
        "end_seconds": {"type": "integer"},
        "assistant_response": {"type": "string"},
        "rejection_reason": {"type": "string"}
    },
    "required": [
        "valid",
        "confidence",
        "goal",
        "current_state",
        "expected_next_state",
        "deviation_type",
        "visual_evidence",
        "could_assist_before_user_realizes",
        "intervention_needed",
        "intervention_timing",
        "start_seconds",
        "deviation_onset_seconds",
        "end_seconds",
        "assistant_response",
        "rejection_reason"
    ]
}


def read_candidates(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_existing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"verifier": "gemini", "records": {}}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def candidate_prompt(row: dict[str, str]) -> str:
    return f"""
You are verifying a dataset item for a benchmark called Streaming Plan Recovery.

Definition:
A valid clip shows a person following an intended plan. At some time, their observed behavior becomes inconsistent with the plan, and a streaming assistant could help them recover before or during the deviation.

Evaluate only the video segment described below. If the URL is a YouTube video and timestamps are present, focus on the requested time range.

Candidate metadata:
- id: {row.get("id", "")}
- title: {row.get("title", "")}
- source_family: {row.get("source_family", "")}
- domain: {row.get("domain", "")}
- candidate_goal_hint: {row.get("goal", "")}
- candidate_start_seconds: {row.get("start_seconds", "")}
- candidate_deviation_onset_seconds: {row.get("deviation_onset_seconds", "") or row.get("timestamp_seconds", "")}
- candidate_end_seconds: {row.get("end_seconds", "")}
- description: {row.get("description", "")}
- transcript/snippet: {row.get("transcript", "")}

Return JSON only.

Rules:
- valid must be true only if the video evidence supports a real plan deviation.
- Do not accept generic "fail" videos unless the intent, deviation, and possible intervention are clear.
- If the metadata is insufficient or the video cannot be accessed, set valid=false and explain rejection_reason.
- Use precise, dataset-friendly labels. Good deviation_type values include:
  missed_target_location, wrong_object, skipped_step, repeated_step, route_failure,
  search_failure, omission, wrong_action, correction, recovery, not_a_deviation.
- Pick the tightest useful segment. Use integers in seconds.
- assistant_response should be what a streaming assistant would say at the intervention moment.
""".strip()


def clip_offsets(row: dict[str, str], default_start: int, default_seconds: int) -> tuple[int, int]:
    start_raw = row.get("start_seconds", "").strip()
    end_raw = row.get("end_seconds", "").strip()
    try:
        start = max(0, int(float(start_raw))) if start_raw else default_start
    except ValueError:
        start = default_start
    try:
        end = max(start + 1, int(float(end_raw))) if end_raw else start + default_seconds
    except ValueError:
        end = start + default_seconds
    return start, end


def request_body(row: dict[str, str], default_start: int, default_seconds: int) -> dict[str, Any]:
    start, end = clip_offsets(row, default_start, default_seconds)
    parts: list[dict[str, Any]] = [{"text": candidate_prompt(row)}]
    url = row.get("url", "").strip()
    if url:
        parts.append(
            {
                "file_data": {"file_uri": url, "mime_type": "video/*"},
                "video_metadata": {
                    "start_offset": f"{start}s",
                    "end_offset": f"{end}s",
                    "fps": 0.5
                }
            }
        )
    return {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.1,
            "response_mime_type": "application/json",
            "response_schema": SCHEMA
        }
    }


def extract_json(response: dict[str, Any]) -> dict[str, Any]:
    try:
        text = response["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Gemini response did not include text: {response}") from exc
    return json.loads(text)


def call_gemini(
    row: dict[str, str],
    api_key: str,
    model: str,
    timeout: int,
    default_start: int,
    default_seconds: int
) -> dict[str, Any]:
    url = API_ENDPOINT.format(model=model)
    payload = json.dumps(request_body(row, default_start, default_seconds)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        },
        method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return extract_json(json.loads(body))


def verify_candidates(args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("Set GEMINI_API_KEY first, or run with --dry-run.")

    candidates = read_candidates(args.candidates)
    output = read_existing(args.out)
    records = output.setdefault("records", {})

    selected = candidates
    if args.limit:
        selected = selected[: args.limit]

    for row in selected:
        record_id = row.get("id", "").strip()
        if not record_id:
            print("Skipping row without id", file=sys.stderr)
            continue
        if record_id in records and not args.force:
            print(f"Skipping {record_id}; already verified")
            continue

        print(f"Verifying {record_id}: {row.get('title', '')}")
        if args.dry_run:
            result = {
                "valid": False,
                "confidence": 0,
                "goal": row.get("goal", ""),
                "current_state": "",
                "expected_next_state": "",
                "deviation_type": "not_verified",
                "visual_evidence": "",
                "could_assist_before_user_realizes": False,
                "intervention_needed": False,
                "intervention_timing": "dry_run",
                "start_seconds": 0,
                "deviation_onset_seconds": 0,
                "end_seconds": 0,
                "assistant_response": "",
                "rejection_reason": "Dry run only; no Gemini API call was made."
            }
        else:
            try:
                result = call_gemini(
                    row,
                    api_key or "",
                    args.model,
                    args.timeout,
                    args.default_start,
                    args.default_clip_seconds
                )
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                result = {
                    "valid": False,
                    "confidence": 0,
                    "goal": row.get("goal", ""),
                    "current_state": "",
                    "expected_next_state": "",
                    "deviation_type": "not_a_deviation",
                    "visual_evidence": "",
                    "could_assist_before_user_realizes": False,
                    "intervention_needed": False,
                    "intervention_timing": "unverified",
                    "start_seconds": 0,
                    "deviation_onset_seconds": 0,
                    "end_seconds": 0,
                    "assistant_response": "",
                    "rejection_reason": f"Gemini API error {exc.code}: {detail}"
                }
                print(f"Rejected {record_id}; Gemini could not process the URL")

        records[record_id] = {
            "model": args.model,
            "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "result": result
        }

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify candidate clips with Gemini video understanding.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--default-start", type=int, default=0)
    parser.add_argument("--default-clip-seconds", type=int, default=90)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_candidates(args)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
