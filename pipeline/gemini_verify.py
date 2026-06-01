from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
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
ALLOWED_DEVIATION_TYPES = [
    "wrong_action",
    "omission",
    "correction",
    "search_failure",
    "route_failure",
    "recovery"
]


SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "valid": {"type": "boolean"},
        "confidence": {"type": "number"},
        "goal": {"type": "string"},
        "current_state": {"type": "string"},
        "expected_next_state": {"type": "string"},
        "deviation_type": {
            "type": "string",
            "enum": ALLOWED_DEVIATION_TYPES + ["not_a_deviation"]
        },
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


def candidate_prompt(row: dict[str, str], clip_start: int, clip_end: int) -> str:
    return f"""
You are a conservative video verifier for a benchmark called Streaming Plan Recovery.
Your job is to reject weak candidates, not to rescue them.

Definition:
A valid clip shows a person following an intended plan. At some time, their observed behavior becomes inconsistent with that plan, and a streaming assistant could help them recover before or during the deviation.

Important:
Do not impose a task structure that is not present in the video or transcript.
Do not infer that ordinary multitasking, rushing, hesitation, or busy work is a mistake.
Do not assume a recipe, route, shopping list, or assembly plan unless the video/transcript/metadata provides enough evidence.
If the evidence does not clearly show a plan deviation, return valid=false.

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
- requested_clip_start_seconds: {clip_start}
- requested_clip_end_seconds: {clip_end}

Return JSON only.

Rules:
- valid must be true only if the video itself or transcript gives direct evidence of a real plan deviation.
- Direct evidence means at least one of:
  1. The user explicitly says they forgot, missed, used the wrong thing, went the wrong way, could not find something, or had to redo/go back.
  2. The video visibly shows the expected target/step/route/object and the user clearly diverges from it.
  3. The transcript plus video together establish the intended plan and the mismatch.
- Reject clips where the only evidence is a title like "crazy rush", "pressure", "chaos", "I was wrong", or "fail" without a visible or spoken deviation.
- Reject clips that only show normal cooking, shopping, walking, searching, serving, waiting, multitasking, or high workload.
- Reject clips if you would need to guess the intended plan, expected next step, or mistake.
- Reject clips if the metadata is insufficient, the video cannot be accessed, or the requested segment does not contain the deviation.
- If valid=true, deviation_type must be exactly one of:
  wrong_action, omission, correction, search_failure, route_failure, recovery.
- If valid=false, deviation_type must be exactly not_a_deviation.
- Do not invent any other deviation_type labels.
- Label definitions:
  wrong_action: direct evidence that the user did the wrong action, used the wrong object, or chose the wrong option.
  omission: direct evidence that the user forgot, missed, skipped, or left out a required target, item, location, or step.
  correction: direct evidence that something must be fixed, redone, corrected, restarted, or tried again.
  search_failure: direct evidence that the user cannot find something they are trying to find.
  route_failure: direct evidence that the user took a wrong turn, missed a turn, got lost, or went the wrong way.
  recovery: direct evidence that the user must go back, turn around, reroute, or backtrack.
- For valid=true:
  current_state must describe only what is visible or explicitly stated.
  expected_next_state must describe only the expected state supported by evidence.
  visual_evidence must name the exact visible/spoken evidence, not a hypothesis.
  confidence should be high only when the evidence is unambiguous.
  assistant_response should be a minimal intervention grounded in the evidence.
- For valid=false:
  current_state, expected_next_state, visual_evidence, assistant_response may be empty.
  rejection_reason must explain what evidence was missing.
- Pick the tightest useful segment only when valid=true. Use integers in seconds.
- Return start_seconds, deviation_onset_seconds, and end_seconds as absolute seconds in the full source video, not seconds relative to the requested segment.
- For valid=true, start_seconds, deviation_onset_seconds, and end_seconds must stay inside the requested clip [{clip_start}, {clip_end}].
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


def youtube_video_id(url: str) -> str:
    for pattern in [
        r"[?&]v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"/shorts/([A-Za-z0-9_-]{11})",
        r"/embed/([A-Za-z0-9_-]{11})"
    ]:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return ""


def is_youtube_url(row: dict[str, str]) -> bool:
    url = row.get("url", "").lower()
    return "youtube.com/" in url or "youtu.be/" in url


def fetch_youtube_duration_seconds(url: str, timeout: int) -> int | None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        page = response.read().decode("utf-8", errors="replace")
    for pattern in [r'"lengthSeconds":"(\d+)"', r'"approxDurationMs":"(\d+)"']:
        match = re.search(pattern, page)
        if match and "Ms" in pattern:
            return max(1, int(match.group(1)) // 1000)
        if match:
            return max(1, int(match.group(1)))
    return None


def explicit_duration(row: dict[str, str]) -> int | None:
    start_raw = row.get("start_seconds", "").strip()
    end_raw = row.get("end_seconds", "").strip()
    if not end_raw:
        return None
    try:
        start = max(0, int(float(start_raw))) if start_raw else 0
        end = max(start + 1, int(float(end_raw)))
        return end
    except ValueError:
        return None


def video_duration(row: dict[str, str], output: dict[str, Any], timeout: int, fallback_seconds: int) -> int:
    durations = output.setdefault("video_durations", {})
    row_id = row.get("id", "").strip()
    if row_id in durations:
        return int(durations[row_id])

    duration = explicit_duration(row)
    if duration is None and is_youtube_url(row):
        try:
            duration = fetch_youtube_duration_seconds(row.get("url", ""), timeout)
        except (TimeoutError, socket.timeout, urllib.error.URLError, urllib.error.HTTPError):
            duration = None
    if duration is None:
        duration = fallback_seconds
    durations[row_id] = duration
    return duration


def clip_id(row_id: str, clip_index: int, start: int, end: int) -> str:
    return f"{row_id}__clip_{clip_index:04d}_{start}_{end}"


def clip_plan(
    candidates: list[dict[str, str]],
    output: dict[str, Any],
    clip_seconds: int,
    duration_timeout: int,
    fallback_video_seconds: int,
    records: dict[str, Any],
    max_new_clips: int,
    force: bool
) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    clip_index = 0
    while True:
        found_clip_in_round = False
        for row in candidates:
            row_id = row.get("id", "").strip()
            if not row_id or not row.get("url", "").strip():
                continue
            duration = video_duration(row, output, duration_timeout, fallback_video_seconds)
            start = clip_index * clip_seconds
            if start >= duration:
                continue
            found_clip_in_round = True
            end = min(duration, start + clip_seconds)
            record_id = clip_id(row_id, clip_index, start, end)
            if record_id in records and not force:
                continue
            ordered.append(
                {
                    "record_id": record_id,
                    "source_candidate_id": row_id,
                    "clip_index": clip_index,
                    "clip_start_seconds": start,
                    "clip_end_seconds": end,
                    "video_duration_seconds": duration,
                    "row": row
                }
            )
            if max_new_clips and len(ordered) >= max_new_clips:
                return ordered
        if not found_clip_in_round:
            break
        clip_index += 1
    return ordered


def normalize_result_timestamps(
    result: dict[str, Any],
    clip_start: int,
    clip_end: int
) -> dict[str, Any]:
    if not result.get("valid"):
        return result

    clip_length = max(0, clip_end - clip_start)
    keys = ("start_seconds", "deviation_onset_seconds", "end_seconds")
    values: list[int] = []
    for key in keys:
        try:
            values.append(int(float(result.get(key))))
        except (TypeError, ValueError):
            return result

    # Gemini sometimes reports timestamps relative to the requested segment.
    # Candidate rows store absolute offsets, so convert clearly relative answers.
    if all(0 <= value <= clip_length for value in values):
        for key, value in zip(keys, values):
            result[key] = min(clip_end, clip_start + value)
    for key in keys:
        try:
            result[key] = min(clip_end, max(clip_start, int(float(result[key]))))
        except (TypeError, ValueError):
            pass
    return result


def request_body(row: dict[str, str], clip_start: int, clip_end: int) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [{"text": candidate_prompt(row, clip_start, clip_end)}]
    url = row.get("url", "").strip()
    if url:
        parts.append(
            {
                "file_data": {"file_uri": url, "mime_type": "video/*"},
                "video_metadata": {
                    "start_offset": f"{clip_start}s",
                    "end_offset": f"{clip_end}s",
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
    clip_start: int,
    clip_end: int
) -> dict[str, Any]:
    url = API_ENDPOINT.format(model=model)
    payload = json.dumps(request_body(row, clip_start, clip_end)).encode("utf-8")
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
    output["clip_seconds"] = args.default_clip_seconds
    output["schedule"] = "round_robin_by_clip_index"
    plan = clip_plan(
        candidates,
        output,
        args.default_clip_seconds,
        args.duration_timeout,
        args.fallback_video_seconds,
        records,
        args.limit,
        args.force
    )

    youtube_seconds_used = 0
    processed = 0
    for clip in plan:
        row = clip["row"]
        record_id = clip["record_id"]
        if not record_id:
            print("Skipping row without id", file=sys.stderr)
            continue
        if record_id in records and not args.force:
            print(f"Skipping {record_id}; already verified")
            continue

        row_clip_seconds = max(0, int(clip["clip_end_seconds"]) - int(clip["clip_start_seconds"]))
        if is_youtube_url(row):
            if youtube_seconds_used + row_clip_seconds > args.max_youtube_seconds_per_run:
                print(
                    "Stopping before daily/run YouTube budget: "
                    f"{youtube_seconds_used}s used, next row needs {row_clip_seconds}s"
                )
                break
            youtube_seconds_used += row_clip_seconds

        print(
            f"Verifying {record_id}: {row.get('title', '')} "
            f"[{clip['clip_start_seconds']}s-{clip['clip_end_seconds']}s]"
        )
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
                "start_seconds": clip["clip_start_seconds"],
                "deviation_onset_seconds": clip["clip_start_seconds"],
                "end_seconds": clip["clip_end_seconds"],
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
                    clip["clip_start_seconds"],
                    clip["clip_end_seconds"]
                )
                result = normalize_result_timestamps(
                    result,
                    clip["clip_start_seconds"],
                    clip["clip_end_seconds"]
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
                    "start_seconds": clip["clip_start_seconds"],
                    "deviation_onset_seconds": clip["clip_start_seconds"],
                    "end_seconds": clip["clip_end_seconds"],
                    "assistant_response": "",
                    "rejection_reason": f"Gemini API error {exc.code}: {detail}"
                }
                print(f"Rejected {record_id}; Gemini could not process the URL")
            except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
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
                    "start_seconds": clip["clip_start_seconds"],
                    "deviation_onset_seconds": clip["clip_start_seconds"],
                    "end_seconds": clip["clip_end_seconds"],
                    "assistant_response": "",
                    "rejection_reason": f"Gemini request failed or timed out: {exc}"
                }
                print(f"Rejected {record_id}; Gemini request failed or timed out")

        records[record_id] = {
            "model": args.model,
            "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_candidate_id": clip["source_candidate_id"],
            "clip_index": clip["clip_index"],
            "clip_start_seconds": clip["clip_start_seconds"],
            "clip_end_seconds": clip["clip_end_seconds"],
            "video_duration_seconds": clip["video_duration_seconds"],
            "candidate": {
                "id": row.get("id", ""),
                "title": row.get("title", ""),
                "url": row.get("url", ""),
                "source_family": row.get("source_family", ""),
                "domain": row.get("domain", ""),
                "goal": row.get("goal", ""),
                "license": row.get("license", ""),
                "license_status": row.get("license_status", ""),
                "hosting_status": row.get("hosting_status", "")
            },
            "result": result
        }
        processed += 1

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

        if args.delay_seconds > 0:
            time.sleep(args.delay_seconds)

        if args.limit and processed >= args.limit:
            print(f"Stopping after {processed} newly processed clips")
            break

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
    parser.add_argument("--limit", type=int, default=20, help="Maximum new clips to verify in this run.")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--default-clip-seconds", type=int, default=300)
    parser.add_argument("--delay-seconds", type=float, default=20)
    parser.add_argument("--max-youtube-seconds-per-run", type=int, default=27000)
    parser.add_argument("--duration-timeout", type=int, default=20)
    parser.add_argument("--fallback-video-seconds", type=int, default=3600)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_candidates(args)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
