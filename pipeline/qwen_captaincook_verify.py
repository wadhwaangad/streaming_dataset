from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "captaincook_qwen_verifications.json"
DEFAULT_VIDEO_CACHE = ROOT / "data" / "captaincook_video_cache"
DEFAULT_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
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


def load_dataset_rows(dataset_name: str, split: str, streaming: bool, decode_videos: bool) -> Any:
    try:
        from datasets import Video, load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Install dependencies first: pip install datasets transformers accelerate torch qwen-vl-utils"
        ) from exc
    rows = load_dataset(dataset_name, split=split, streaming=streaming)
    if decode_videos:
        return rows

    features = getattr(rows, "features", None) or {}
    for column_name, feature in features.items():
        if isinstance(feature, Video):
            rows = rows.cast_column(column_name, Video(decode=False))
    return rows


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


def download_hf_video(uri: str, cache_dir: Path) -> str:
    match = re.match(r"^hf://datasets/([^@]+)@([^/]+)/(.+)$", uri)
    if not match:
        return uri
    repo_id, revision, filename = match.groups()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: pip install huggingface_hub") from exc
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        repo_type="dataset",
        cache_dir=str(cache_dir / "hf"),
    )


def materialize_video(value: Any, cache_dir: Path, record_id: str) -> str:
    path = video_path(value)
    if path:
        return download_hf_video(path, cache_dir)
    if not isinstance(value, dict) or not value.get("bytes"):
        return ""
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(str(value.get("path") or "clip.mp4")).suffix or ".mp4"
    out_path = cache_dir / f"{record_id}{suffix}"
    if not out_path.exists():
        out_path.write_bytes(value["bytes"])
    return str(out_path)


def local_video_duration_seconds(path: str) -> int | None:
    if not path:
        return None
    try:
        import decord
    except ImportError:
        return None
    try:
        reader = decord.VideoReader(path)
        fps = float(reader.get_avg_fps())
        if fps <= 0:
            return None
        return max(1, int(len(reader) / fps))
    except Exception:
        return None


def scan_windows(
    local_video: str,
    start_seconds: int,
    end_seconds: int,
    window_seconds: int,
    stride_seconds: int,
    max_windows: int,
) -> list[tuple[int, int]]:
    duration = local_video_duration_seconds(local_video)
    scan_end = duration if duration is not None else end_seconds
    scan_end = max(start_seconds + 1, scan_end)
    window_seconds = max(1, window_seconds)
    stride_seconds = max(1, stride_seconds)

    windows: list[tuple[int, int]] = []
    cursor = start_seconds
    while cursor < scan_end:
        window_end = min(scan_end, cursor + window_seconds)
        windows.append((cursor, window_end))
        if max_windows and len(windows) >= max_windows:
            break
        if window_end >= scan_end:
            break
        cursor += stride_seconds
    return windows


def prompt_for_row(row: dict[str, Any], start_seconds: int, end_seconds: int, raw_candidates: bool) -> str:
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

Mode:
- raw_unlabeled_candidates: {str(raw_candidates).lower()}

Return JSON only with these keys:
valid, confidence, goal, current_state, expected_next_state, deviation_type, visual_evidence, could_assist_before_user_realizes, intervention_needed, intervention_timing, start_seconds, deviation_onset_seconds, end_seconds, assistant_response, rejection_reason.

Rules:
- valid must be true only if the video plus metadata gives direct evidence of a real procedural deviation.
- You must identify the intended next step from recipe/task, narration/transcript, error/mistake label, or step/action annotations.
- If raw_unlabeled_candidates=false and recipe/task, narration/transcript, error/mistake label, and step/action annotations are all empty, return valid=false.
- Reject if the expected_next_state is not explicitly supported by metadata or visible/spoken evidence.
- Reject if you are guessing what the cook should do next from common cooking knowledge.
- Reject if the only issue is that the person is doing a normal preparation action, such as peeling, chopping, stirring, washing, pouring, waiting, or moving ingredients.
- Example rejection: if the clip shows peeling garlic but no recipe, no instruction to add garlic to a pot, and no mistake label, return valid=false because peeling garlic may be a correct step.
- In raw_unlabeled_candidates mode, valid=true requires obvious self-contained evidence inside the clip itself, such as a visible spill/drop/burn/breakage, unsafe action, failed object state, reversal, redo, visible correction, or on-screen/speaker cue that something went wrong.
- In raw_unlabeled_candidates mode, reject normal cooking unless the failure or correction is visible without knowing the recipe.
- In raw_unlabeled_candidates mode, do not use "instead of", "should", or "expected" unless the clip itself visibly establishes the intended alternative.
- In raw_unlabeled_candidates mode, set confidence below 0.7 unless the mistake is unmistakable to a viewer with no recipe context.
- If valid=false, deviation_type must be "not_a_deviation".
- If valid=true, deviation_type must be one of: wrong_action, omission, correction, search_failure, route_failure, recovery.
- Do not treat ordinary cooking, waiting, multitasking, prep work, or a correct step as a deviation.
- For valid=true, visual_evidence must cite the grounded mismatch: "metadata says X should happen, but the video shows Y" or "speaker says X was wrong/forgotten".
- For valid=true, confidence must be below 0.7 unless there is an explicit mistake/error label, narration cue, or step annotation conflict.
- Timestamps must be absolute seconds in the source video and must stay inside the requested clip.
- assistant_response should be a short intervention grounded in the evidence.
""".strip()


def has_grounding_metadata(row: dict[str, Any]) -> bool:
    return any(
        text_value(row, keys).strip()
        for keys in (
            ("recipe", "recipe_name", "task", "activity", "goal"),
            ("narration", "transcript", "description", "caption"),
            ("error", "error_type", "mistake", "label", "annotation"),
            ("steps", "step_annotations", "actions", "action_annotations"),
        )
    )


def raw_result_is_grounded(result: dict[str, Any]) -> bool:
    if not result.get("valid"):
        return True
    evidence = " ".join(
        str(result.get(key) or "").lower()
        for key in ("visual_evidence", "current_state", "expected_next_state", "assistant_response")
    )
    speculative = (
        "instead of",
        "should",
        "expected",
        "supposed to",
        "recipe",
        "add it",
        "add the",
        "mixing ingredients",
    )
    concrete_failure = (
        "spill",
        "spilled",
        "drop",
        "dropped",
        "fall",
        "fell",
        "burn",
        "burnt",
        "overflow",
        "break",
        "broke",
        "broken",
        "wrong way",
        "wrong",
        "mistake",
        "forgot",
        "missed",
        "redo",
        "fix",
        "correct",
        "restart",
        "start over",
        "go back",
        "return",
        "revers",
        "corrects",
        "correction",
    )
    return any(term in evidence for term in concrete_failure) and not any(term in evidence for term in speculative)


def reject_ungrounded_raw_result(result: dict[str, Any], start_seconds: int, end_seconds: int) -> dict[str, Any]:
    if raw_result_is_grounded(result):
        return result
    return {
        "valid": False,
        "confidence": 0,
        "goal": "",
        "current_state": "",
        "expected_next_state": "",
        "deviation_type": "not_a_deviation",
        "visual_evidence": "",
        "could_assist_before_user_realizes": False,
        "intervention_needed": False,
        "intervention_timing": "unverified",
        "start_seconds": start_seconds,
        "deviation_onset_seconds": start_seconds,
        "end_seconds": end_seconds,
        "assistant_response": "",
        "rejection_reason": (
            "Rejected after model call because this raw unlabeled candidate did not provide "
            "self-contained visible evidence of an actual failure, correction, or mistake."
        ),
    }


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


def load_qwen(model_name: str, dtype_name: str) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise SystemExit(
            "Install dependencies first: pip install transformers accelerate torch qwen-vl-utils"
        ) from exc

    device_map = "auto" if torch.cuda.is_available() else None
    if dtype_name == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    elif dtype_name == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_name == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    if not torch.cuda.is_available():
        dtype = torch.float32
    print(f"Loading {model_name} with device_map={device_map or 'cpu'} dtype={dtype}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype, device_map=device_map)
    processor = AutoProcessor.from_pretrained(model_name)
    if device_map is None:
        model = model.to("cpu")
    model.eval()
    return model, processor


def torch_dtype(dtype_name: str) -> Any:
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    if dtype_name == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


def load_smolvlm(model_name: str, dtype_name: str) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install transformers accelerate torch decord pillow") from exc

    dtype = torch_dtype(dtype_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device} dtype={dtype}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    return model, processor


def model_backend(model_name: str) -> str:
    lowered = model_name.lower()
    if "smolvlm" in lowered:
        return "smolvlm"
    return "qwen"


def load_local_model(model_name: str, dtype_name: str) -> tuple[Any, Any, str]:
    backend = model_backend(model_name)
    if backend == "smolvlm":
        model, processor = load_smolvlm(model_name, dtype_name)
    else:
        model, processor = load_qwen(model_name, dtype_name)
    return model, processor, backend


def validate_video_reader(video_reader: str) -> None:
    if video_reader != "auto":
        os.environ["FORCE_QWENVL_VIDEO_READER"] = video_reader

    has_decord = importlib.util.find_spec("decord") is not None
    has_torchcodec = importlib.util.find_spec("torchcodec") is not None
    if video_reader in {"decord", "auto"} and has_decord:
        return
    if video_reader in {"torchcodec", "auto"} and has_torchcodec:
        return

    try:
        from torchvision import io
    except ImportError:
        io = None
    if video_reader in {"torchvision", "auto"} and io is not None and hasattr(io, "read_video"):
        return

    if video_reader == "decord":
        raise SystemExit("The decord video reader is not installed. Run: pip install decord")
    if video_reader == "torchcodec":
        raise SystemExit("The torchcodec video reader is not installed. Run: pip install torchcodec")
    if video_reader == "torchvision":
        raise SystemExit(
            "Your torchvision build does not provide torchvision.io.read_video. "
            "Use --video-reader decord after running: pip install decord"
        )
    raise SystemExit(
        "No working Qwen video reader is available. Run this once, then retry:\n"
        "pip install decord\n\n"
        "Then run:\n"
        "python -m pipeline.qwen_captaincook_verify --limit 5 --streaming --video-reader decord"
    )


def call_qwen(
    model: Any,
    processor: Any,
    model_name: str,
    video: Any,
    prompt: str,
    max_new_tokens: int,
    video_fps: float,
    video_nframes: int,
    video_max_pixels: int,
    video_start: int,
    video_end: int,
) -> dict[str, Any]:
    import torch

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise SystemExit("Install qwen-vl-utils before running the local video verifier.") from exc

    content: list[dict[str, Any]] = []
    path = video_path(video)
    if path:
        video_part: dict[str, Any] = {
            "type": "video",
            "video": path,
            "video_start": float(video_start),
            "video_end": float(video_end),
            "min_pixels": 100352,
            "max_pixels": max(video_max_pixels, 100352),
        }
        if video_nframes > 0:
            video_part["nframes"] = video_nframes
        else:
            video_part["fps"] = video_fps
            video_part["max_frames"] = 16
        content.append(video_part)
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.inference_mode():
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


def sample_video_frames(path: str, start_seconds: int, end_seconds: int, frame_count: int) -> list[Any]:
    try:
        import decord
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Install video frame dependencies first: pip install decord pillow numpy") from exc

    reader = decord.VideoReader(path)
    fps = float(reader.get_avg_fps())
    total_frames = len(reader)
    if total_frames <= 0 or fps <= 0:
        return []
    start_index = max(0, min(total_frames - 1, int(start_seconds * fps)))
    end_index = max(start_index + 1, min(total_frames, int(end_seconds * fps)))
    count = max(1, frame_count)
    indices = np.linspace(start_index, end_index - 1, num=min(count, end_index - start_index), dtype=int)
    batch = reader.get_batch(indices.tolist()).asnumpy()
    return [Image.fromarray(frame).convert("RGB") for frame in batch]


def call_smolvlm(
    model: Any,
    processor: Any,
    model_name: str,
    video: Any,
    prompt: str,
    max_new_tokens: int,
    video_nframes: int,
    video_start: int,
    video_end: int,
) -> dict[str, Any]:
    import torch

    path = video_path(video)
    frames = sample_video_frames(path, video_start, video_end, video_nframes)
    if not frames:
        raise ValueError(f"No frames could be sampled from {path}")

    content = [{"type": "image"} for _ in frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=frames, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = generated_ids[:, inputs["input_ids"].shape[-1] :]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    result = extract_json(answer)
    result["raw_model_response"] = answer
    result["model"] = model_name
    result["sampled_frames"] = len(frames)
    result["sampled_window"] = {"start_seconds": video_start, "end_seconds": video_end}
    return result


def call_verifier_model(
    backend: str,
    model: Any,
    processor: Any,
    model_name: str,
    video: Any,
    prompt: str,
    max_new_tokens: int,
    video_fps: float,
    video_nframes: int,
    video_max_pixels: int,
    video_start: int,
    video_end: int,
) -> dict[str, Any]:
    if backend == "smolvlm":
        return call_smolvlm(
            model,
            processor,
            model_name,
            video,
            prompt,
            max_new_tokens,
            video_nframes,
            video_start,
            video_end,
        )
    return call_qwen(
        model,
        processor,
        model_name,
        video,
        prompt,
        max_new_tokens,
        video_fps,
        video_nframes,
        video_max_pixels,
        video_start,
        video_end,
    )


def cleanup_gpu_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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

    rows = load_dataset_rows(args.dataset, args.split, args.streaming, args.decode_videos)
    model = processor = backend = None
    if not args.dry_run:
        backend = model_backend(args.model)
        if backend == "qwen":
            validate_video_reader(args.video_reader)
        model, processor, backend = load_local_model(args.model, args.dtype)

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
        if not local_video and not args.dry_run:
            raise SystemExit(f"No video file/path found for {record_id}; row keys: {sorted(row.keys())}")
        if args.dry_run:
            result = dry_result(row, start_seconds, end_seconds)
        elif args.require_metadata and not args.raw_candidates and not has_grounding_metadata(row):
            result = {
                "valid": False,
                "confidence": 0,
                "goal": "",
                "current_state": "",
                "expected_next_state": "",
                "deviation_type": "not_a_deviation",
                "visual_evidence": "",
                "could_assist_before_user_realizes": False,
                "intervention_needed": False,
                "intervention_timing": "unverified",
                "start_seconds": start_seconds,
                "deviation_onset_seconds": start_seconds,
                "end_seconds": end_seconds,
                "assistant_response": "",
                "rejection_reason": (
                    "Rejected before model call because this row has no recipe/task, narration/transcript, "
                    "mistake label, or step/action annotations to ground an expected next step."
                ),
            }
        else:
            result = call_verifier_model(
                backend,
                model,
                processor,
                args.model,
                local_video,
                prompt_for_row(row, start_seconds, end_seconds, args.raw_candidates),
                args.max_new_tokens,
                args.video_fps,
                args.video_nframes,
                args.video_max_pixels,
                start_seconds,
                end_seconds,
            )
            if args.raw_candidates and not has_grounding_metadata(row):
                result = reject_ungrounded_raw_result(result, start_seconds, end_seconds)
        result = normalize_result(result, start_seconds, end_seconds)
        cleanup_gpu_cache()

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
    parser = argparse.ArgumentParser(description="Verify CaptainCook4D clips with a local video-language model.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--default-clip-seconds", type=int, default=60)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--video-cache", type=Path, default=DEFAULT_VIDEO_CACHE)
    parser.add_argument("--video-reader", choices=("auto", "decord", "torchcodec", "torchvision"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--video-fps", type=float, default=0.25)
    parser.add_argument("--video-nframes", type=int, default=32)
    parser.add_argument("--video-max-pixels", type=int, default=100352)
    parser.add_argument("--decode-videos", action="store_true")
    parser.add_argument("--raw-candidates", action="store_true")
    parser.add_argument("--allow-ungrounded", dest="require_metadata", action="store_false")
    parser.set_defaults(require_metadata=True)
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
