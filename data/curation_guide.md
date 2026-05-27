# Dataset Curation Guide

Use this workflow to turn source pointers and search results into a VLM-verified clip-level dataset.

## 1. Find Candidate Videos

Start from the source pointers in `data/source_map.json` or from web searches using cue terms such as:

- `forgot`, `missed`, `skipped`, `wrong`, `mistake`
- `couldn't find`, `wrong turn`, `got lost`
- `had to go back`, `turn around`, `redo`, `fix`

For every candidate, save the original URL and enough context to review it later.

You can also add channels to `data/channels.csv` and run:

```powershell
python -m pipeline.collect_candidates --channels data/channels.csv --candidates data/candidates.csv
```

The collector reads public channel RSS feeds and appends new uploads that match deviation cue terms. Set `include_all` to `true` for broad channels where you want Gemini to inspect everything.

## 2. Queue Raw Candidates

Each row in `data/candidates.csv` should describe one candidate video or segment. These fields are hints for the verifier, not final labels.

Useful timing hints:

- `start_seconds`: usually 10-30 seconds before the deviation.
- `deviation_onset_seconds`: the first moment behavior visibly diverges from the plan.
- `end_seconds`: usually 5-30 seconds after the recovery, correction, or failure becomes clear.

Prefer segments that are short enough to inspect quickly but long enough to show intent, deviation, and recovery.

## 3. Verify With Gemini

Run:

```powershell
python -m pipeline.gemini_verify --candidates data/candidates.csv --out data/verifications.json
python -m pipeline.run_pipeline --candidates data/candidates.csv --verifications data/verifications.json --out site/data/dataset.json --report build/report.md
```

Gemini marks a clip as `curated` only when the clip visibly satisfies all of these:

- The user has an intended goal or plan.
- The observed behavior diverges from that plan.
- The deviation is visible or strongly grounded in transcript plus video.
- A streaming assistant could plausibly help before or during the mistake.

Unverified clips remain `awaiting_vlm`. Clips that use mistake language but do not show a useful intervention moment become `rejected`.

## 4. Record Rights Clearly

Use `hosting_status` to say what your site can do:

- `external_link_only`: safest default for YouTube, papers, and most third-party datasets.
- `metadata_only`: use when even deep-linking clips is questionable, but source citation is useful.
- `hosted_allowed`: only when redistribution is clearly permitted.

Use `license_status`:

- `redistribution_allowed`
- `external_link_only`
- `unknown`
- `restricted`

When in doubt, link to the original source and do not upload the video file.

## 5. Publish

After editing `data/candidates.csv`, run:

```powershell
python -m pipeline.run_pipeline --candidates data/candidates.csv --out site/data/dataset.json --report build/report.md
```

Then commit and push. GitHub Actions will rebuild and publish the site.
