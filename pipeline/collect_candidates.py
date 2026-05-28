from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHANNELS = ROOT / "data" / "channels.csv"
DEFAULT_CANDIDATES = ROOT / "data" / "candidates.csv"
DEFAULT_SOURCE_MAP = ROOT / "data" / "source_map.json"

CSV_FIELDS = [
    "id",
    "title",
    "description",
    "transcript",
    "url",
    "source_family",
    "domain",
    "goal",
    "video_duration_seconds",
    "start_seconds",
    "deviation_onset_seconds",
    "end_seconds",
    "license",
    "license_status",
    "hosting_status",
    "notes"
]

ATOM = "{http://www.w3.org/2005/Atom}"
YT = "{http://www.youtube.com/xml/schemas/2015}"
MEDIA = "{http://search.yahoo.com/mrss/}"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_candidates(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def feed_url(channel: dict[str, str]) -> str:
    explicit = channel.get("feed_url", "").strip()
    if explicit:
        return explicit
    channel_id = channel.get("channel_id", "").strip()
    channel_url = channel.get("channel_url", "").strip()
    if not channel_id and channel_url:
        channel_id = resolve_channel_id(channel_url)
    if not channel_id:
        raise ValueError("Channel row needs feed_url, channel_id, or channel_url")
    return f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"


def resolve_channel_id(channel_url: str) -> str:
    match = re.search(r"/channel/(UC[\w-]+)", channel_url)
    if match:
        return match.group(1)

    request = urllib.request.Request(channel_url, headers={"User-Agent": "streaming-dataset-candidate-collector/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", errors="replace")

    for pattern in [
        r'"channelId":"(UC[\w-]+)"',
        r'"externalId":"(UC[\w-]+)"',
        r'<meta itemprop="channelId" content="(UC[\w-]+)"'
    ]:
        match = re.search(pattern, html)
        if match:
            return match.group(1)
    raise ValueError(f"Could not resolve YouTube channel ID from {channel_url}")


def fetch_feed(url: str, timeout: int) -> ET.Element:
    request = urllib.request.Request(url, headers={"User-Agent": "streaming-dataset-candidate-collector/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return ET.fromstring(body)


def fetch_channel_videos(channel_url: str, timeout: int) -> list[dict[str, str]]:
    videos_url = channel_url.rstrip("/") + "/videos"
    request = urllib.request.Request(videos_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        page = response.read().decode("utf-8", errors="replace")

    videos: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in re.finditer(r'"videoId":"([A-Za-z0-9_-]{11})"', page):
        video_id = match.group(1)
        if video_id in seen:
            continue
        seen.add(video_id)

        window = page[match.start() : match.start() + 3500]
        title = ""
        title_match = re.search(r'"content":"([^"]+)"', window) or re.search(r'"text":"([^"]+)"', window)
        if title_match:
            title = repair_mojibake(html.unescape(title_match.group(1).replace(r"\u0026", "&")))
        duration_seconds = page_duration_seconds(window)

        videos.append(
            {
                "video_id": video_id,
                "title": title,
                "description": "",
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "published": "",
                "duration_seconds": str(duration_seconds) if duration_seconds is not None else ""
            }
        )
    return videos


def text(node: ET.Element | None) -> str:
    return node.text.strip() if node is not None and node.text else ""


def parse_seconds(value: str | None) -> int | None:
    if not value:
        return None
    try:
        seconds = int(float(value.strip()))
    except ValueError:
        return None
    return seconds if seconds > 0 else None


def page_duration_seconds(value: str) -> int | None:
    match = re.search(r'"lengthSeconds":"(\d+)"', value)
    if match:
        return parse_seconds(match.group(1))

    match = re.search(r'"lengthText":\{"accessibility":\{"accessibilityData":\{"label":"([^"]+)"', value)
    if not match:
        match = re.search(r'"lengthText":\{"simpleText":"([^"]+)"', value)
    if not match:
        return None

    parts = [int(part) for part in re.findall(r"\d+", match.group(1))]
    if not parts:
        return None
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds or None


def repair_mojibake(value: str) -> str:
    if not value or not any(marker in value for marker in ("Ã", "Â", "â", "ð")):
        return value
    try:
        return value.encode("latin1").decode("utf-8")
    except UnicodeError:
        return value


def entries(root: ET.Element) -> list[dict[str, str]]:
    rows = []
    for entry in root.findall(f"{ATOM}entry"):
        video_id = text(entry.find(f"{YT}videoId"))
        title = repair_mojibake(text(entry.find(f"{ATOM}title")))
        link = entry.find(f"{ATOM}link")
        url = link.attrib.get("href", "") if link is not None else ""
        group = entry.find(f"{MEDIA}group")
        description = text(group.find(f"{MEDIA}description")) if group is not None else ""
        duration_seconds = ""
        if group is not None:
            duration_node = group.find(f"{YT}duration")
            duration_seconds = (
                duration_node.attrib.get("seconds", "").strip()
                if duration_node is not None
                else ""
            )
        published = text(entry.find(f"{ATOM}published"))
        rows.append(
            {
                "video_id": video_id,
                "title": title,
                "description": description,
                "url": url,
                "published": published,
                "duration_seconds": duration_seconds
            }
        )
    return rows


def cue_terms(source_map: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for phrases in source_map.get("deviation_cues", {}).values():
        terms.extend(phrases)
    extras = ["fail", "failure", "went wrong", "oops", "forgotten", "lost", "back to the store"]
    return sorted(set(term.lower() for term in terms + extras))


def looks_relevant(video: dict[str, str], terms: list[str]) -> bool:
    haystack = f"{video.get('title', '')} {video.get('description', '')}".lower()
    return any(term in haystack for term in terms)


def candidate_id(video: dict[str, str]) -> str:
    video_id = video.get("video_id", "").strip()
    if video_id:
        return "yt_" + video_id
    digest = hashlib.sha1(video.get("url", "").encode("utf-8")).hexdigest()[:10]
    return "url_" + digest


def clip_id(base_id: str, start_seconds: int, end_seconds: int) -> str:
    return f"{base_id}_clip{start_seconds:06d}_{end_seconds:06d}"


def clip_windows(duration_seconds: int | None, clip_seconds: int) -> list[tuple[int, int]]:
    if duration_seconds is None:
        return [(0, clip_seconds)]
    full_clip_count = duration_seconds // clip_seconds
    return [
        (index * clip_seconds, (index + 1) * clip_seconds)
        for index in range(full_clip_count)
    ]


def build_candidate(
    video: dict[str, str],
    channel: dict[str, str],
    start_seconds: int,
    end_seconds: int,
    duration_seconds: int | None
) -> dict[str, str]:
    base_id = candidate_id(video)
    return {
        "id": clip_id(base_id, start_seconds, end_seconds),
        "title": video.get("title", ""),
        "description": video.get("description", ""),
        "transcript": "",
        "url": video.get("url", ""),
        "source_family": channel.get("source_family", "").strip() or "YouTube/channel",
        "domain": channel.get("domain", "").strip() or "unknown",
        "goal": channel.get("goal_hint", "").strip(),
        "video_duration_seconds": str(duration_seconds) if duration_seconds is not None else "",
        "start_seconds": str(start_seconds),
        "deviation_onset_seconds": "",
        "end_seconds": str(end_seconds),
        "license": "review_required",
        "license_status": "unknown",
        "hosting_status": "external_link_only",
        "notes": "Auto-collected fixed-length clip from channel feed; VLM must verify."
    }


def collect(args: argparse.Namespace) -> tuple[int, int]:
    if args.clip_seconds <= 0:
        raise SystemExit("--clip-seconds must be greater than 0.")

    channels = read_csv(args.channels)
    candidates = read_csv(args.candidates)
    existing_by_id = {row.get("id", "").strip(): row for row in candidates}
    terms = cue_terms(read_json(args.source_map))
    added = 0
    refreshed = 0
    seen = 0

    for channel in channels:
        if channel.get("channel_id", "").startswith("UC_EXAMPLE"):
            continue
        include_all = channel.get("include_all", "").strip().lower() in {"true", "1", "yes"}
        try:
            root = fetch_feed(feed_url(channel), args.timeout)
            videos = entries(root)
        except urllib.error.HTTPError as exc:
            channel_url = channel.get("channel_url", "").strip()
            if exc.code not in {404, 500, 502, 503, 504} or not channel_url:
                raise
            videos = fetch_channel_videos(channel_url, args.timeout)

        for video in videos[: args.max_per_channel]:
            seen += 1
            duration_seconds = parse_seconds(video.get("duration_seconds", ""))
            rows = [
                build_candidate(video, channel, start, end, duration_seconds)
                for start, end in clip_windows(duration_seconds, args.clip_seconds)
            ]
            if not rows:
                continue
            if not include_all and not looks_relevant(video, terms):
                continue
            for row in rows:
                existing = existing_by_id.get(row["id"])
                if existing:
                    if existing.get("notes", "").startswith("Auto-collected"):
                        existing["title"] = row["title"] or existing.get("title", "")
                        existing["description"] = row["description"] or existing.get("description", "")
                        existing["video_duration_seconds"] = (
                            row["video_duration_seconds"] or existing.get("video_duration_seconds", "")
                        )
                        existing["start_seconds"] = row["start_seconds"]
                        existing["end_seconds"] = row["end_seconds"]
                        refreshed += 1
                    continue
                candidates.append(row)
                existing_by_id[row["id"]] = row
                added += 1

    if not args.dry_run:
        write_candidates(args.candidates, candidates)
    return seen, added, refreshed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect raw candidate URLs from YouTube channel RSS feeds.")
    parser.add_argument("--channels", type=Path, default=DEFAULT_CHANNELS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--source-map", type=Path, default=DEFAULT_SOURCE_MAP)
    parser.add_argument("--max-per-channel", type=int, default=15)
    parser.add_argument("--clip-seconds", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seen, added, refreshed = collect(args)
    action = "Would add" if args.dry_run else "Added"
    print(f"Checked {seen} recent videos")
    print(f"{action} {added} new candidate rows")
    print(f"Refreshed {refreshed} existing auto-collected rows")
    print(f"Finished at {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
