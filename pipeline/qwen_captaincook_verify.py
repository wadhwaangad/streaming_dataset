from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "captaincook_qwen_verifications.json"
DEFAULT_VIDEO_CACHE = ROOT / "data" / "captaincook_video_cache"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_DATASET = "SabrianLinnn/captain_cook_4d"

ALLOWED_DEVIATION_TYPES = {
    "wrong_action",
    "omission",
    "correction",
    "search_failure",
    "route_failure",
    "recovery",
    "not_a_deviation",
}


def read_existing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"verifier": "qwen2.5-vl-local", "dataset": DEFAULT_DATASET, "records": {}}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_dataset_rows(dataset_name: str, split: str, streaming: bool) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Install dependencies first: pip install datasets transformers accelerate torch qwen-vl-utils"
        ) from exc
    return load_dataset(dataset_name, split=split, streaming=streaming)


def row_id(row: dict[str, Any], index: int) -> str:
    for key in ("id", "sample_id", "video_id", "recording_id", "uid"):
        value = row.get(key)
        if value:
            return str(value)
    return f"captaincook_{index:06d}"


def text_value(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)
    return ""


def seconds_value(row: dict[str, Any], keys: tuple[str, ...], fallback: int) -> int:
    for key in keys:
        value = row.get(key)
        if value is None or value == "":
            continue
        try:
            return max(0, int(float(value)))
        except (TypeError, ValueError):
            continue
    return fallback


def video_value(row: dict[str, Any]) -> Any:
    for key in ("video", "video_path", "video_file", "path", "file", "mp4"):
        value = row.get(key)
        if value:
            return value
    return None


def video_path(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("path", "filename", "url"):
            if value.get(key):
                return str(value[key])
    return ""


def materialize_video(value: Any, cache_dir: Path, record_id: str) -> str:
    path = video_path(value)
    if path:
        return path
    if not isinstance(value, dict) or not value.get("bytes"):
        return ""
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(str(value.get("path") or "clip.mp4")).suffix or ".mp4"
    out_path = cache_dir / f"{record_id}{suffix}"
    if not out_path.exists():
        out_path.write_bytes(value["bytes"])
    return str(out_path)


def prompt_for_row(row: dict[str, Any], start_seconds: int, end_seconds: int) -> str:
    recipe = text_value(row, ("recipe", "recipe_name", "task", "activity", "goal"))
    narration = text_value(row, ("narration", "transcript", "description", "caption"))
    error = text_value(row, ("error", "error_type", "mistake", "label", "annotation"))
    steps = text_value(row, ("steps", "step_annotations", "actions", "action_annotations"))
    return f"""
You are a conservative verifier for the Streaming Plan Recovery benchmark.

Definition:
A valid clip shows a person following an intended cooking procedure. Their observed behavior becomes inconsistent with the intended recipe or step sequence, and a streaming assistant could help them recover before or during the mistake.

CaptainCook4D metadata:
- recipe/task: {recipe}
- narration/transcript: {narration}
- error/mistake label: {error}
- step/action annotations: {steps}
- requested_clip_start_seconds: {start_seconds}
- requested_clip_end_seconds: {end_seconds}

Return JSON only with these keys:
valid, confidence, goal, current_state, expected_next_state, deviation_type, visual_evidence, could_assist_before_user_realizes, intervention_needed, intervention_timing, start_seconds, deviation_onset_seconds, end_seconds, assistant_response, rejection_reason.

Rules:
- valid must be true only if the video or metadata gives direct evidence of a real procedural deviation.
- If valid=false, deviation_type must be "not_a_deviation".
- If valid=true, deviation_type must be one of: wrong_action, omission, correction, search_failure, route_failure, recovery.
- Do not treat ordinary cooking, waiting, multitasking, or a correct step as a deviation.
- Timestamps must be absolute seconds in the source video and must stay inside the requested clip.
- assistant_response should be a short intervention grounded in the evidence.
""".strip()


def extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_result(result: dict[str, Any], start_seconds: int, end_seconds: int) -> dict[str, Any]:
    valid = bool(result.get("valid"))
    result["valid"] = valid
    deviation_type = str(result.get("deviation_type") or "").strip()
    if deviation_type not in ALLOWED_DEVIATION_TYPES:
        deviation_type = "not_a_deviation" if not valid else "wrong_action"
    if not valid:
        deviation_type = "not_a_deviation"
    result["deviation_type"] = deviation_type

    for key, fallback in (
        ("confidence", 0.0),
        ("goal", ""),
        ("current_state", ""),
        ("expected_next_state", ""),
        ("visual_evidence", ""),
        ("intervention_timing", "unverified"),
        ("assistant_response", ""),
        ("rejection_reason", ""),
    ):
        result.setdefault(key, fallback)
    for key, fallback in (("could_assist_before_user_realizes", False), ("intervention_needed", False)):
        result[key] = bool(result.get(key, fallback))
    for key, fallback in (
        ("start_seconds", start_seconds),
        ("deviation_onset_seconds", start_seconds),
        ("end_seconds", end_seconds),
    ):
        try:
            result[key] = min(end_seconds, max(start_seconds, int(float(result.get(key, fallback)))))
        except (TypeError, ValueError):
            result[key] = fallback
    return result


def load_qwen(model_name: str) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise SystemExit(
            "Install dependencies first: pip install transformers accelerate torch qwen-vl-utils"
        ) from exc

    device_map = "auto" if torch.cuda.is_available() else None
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype, device_map=device_map)
    processor = AutoProcessor.from_pretrained(model_name)
    if device_map is None:
        model = model.to("cpu")
    model.eval()
    return model, processor


def call_qwen(
    model: Any,
    processor: Any,
    model_name: str,
    video: Any,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise SystemExit("Install qwen-vl-utils before running the local video verifier.") from exc

    content: list[dict[str, Any]] = []
    path = video_path(video)
    if path:
        content.append({"type": "video", "video": path, "fps": 1.0})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    result = extract_json(answer)
    result["raw_model_response"] = answer
    result["model"] = model_name
    return result


def dry_result(row: dict[str, Any], start_seconds: int, end_seconds: int) -> dict[str, Any]:
    return {
        "valid": False,
        "confidence": 0,
        "goal": text_value(row, ("recipe", "recipe_name", "task", "activity", "goal")),
        "current_state": "",
        "expected_next_state": "",
        "deviation_type": "not_a_deviation",
        "visual_evidence": "",
        "could_assist_before_user_realizes": False,
        "intervention_needed": False,
        "intervention_timing": "dry_run",
        "start_seconds": start_seconds,
        "deviation_onset_seconds": start_seconds,
        "end_seconds": end_seconds,
        "assistant_response": "",
        "rejection_reason": "Dry run only; no local Qwen model was loaded.",
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    output = read_existing(args.out)
    output["dataset"] = args.dataset
    output["model"] = args.model
    records = output.setdefault("records", {})

    rows = load_dataset_rows(args.dataset, args.split, args.streaming)
    model = processor = None
    if not args.dry_run:
        model, processor = load_qwen(args.model)

    processed = 0
    for index, row in enumerate(rows):
        if args.skip and index < args.skip:
            continue
        if args.limit and processed >= args.limit:
            break
        row = dict(row)
        source_id = row_id(row, index)
        start_seconds = seconds_value(row, ("start_seconds", "start", "clip_start", "timestamp_start"), 0)
        end_seconds = seconds_value(
            row,
            ("end_seconds", "end", "clip_end", "timestamp_end"),
            start_seconds + args.default_clip_seconds,
        )
        record_id = f"{source_id}__{start_seconds}_{end_seconds}"
        if record_id in records and not args.force:
            continue

        print(f"Verifying {record_id}")
        local_video = materialize_video(video_value(row), args.video_cache, record_id)
        if args.dry_run:
            result = dry_result(row, start_seconds, end_seconds)
        else:
            result = call_qwen(
                model,
                processor,
                args.model,
                local_video,
                prompt_for_row(row, start_seconds, end_seconds),
                args.max_new_tokens,
            )
        result = normalize_result(result, start_seconds, end_seconds)

        records[record_id] = {
            "model": args.model,
            "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset": args.dataset,
            "split": args.split,
            "source_candidate_id": source_id,
            "clip_start_seconds": start_seconds,
            "clip_end_seconds": end_seconds,
            "source_row": {
                "id": source_id,
                "recipe": text_value(row, ("recipe", "recipe_name", "task", "activity", "goal")),
                "label": text_value(row, ("error", "error_type", "mistake", "label", "annotation")),
                "video": local_video,
            },
            "result": result,
        }
        write_json(args.out, output)
        processed += 1

    write_json(args.out, output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify CaptainCook4D clips with local Qwen2.5-VL-3B.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--default-clip-seconds", type=int, default=60)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--video-cache", type=Path, default=DEFAULT_VIDEO_CACHE)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify(args)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
