# Trajectory Deviation Dataset Pipeline

This repository contains a small, reproducible pipeline for building a trajectory-deviation video dataset and a static website for browsing the resulting manifest.

The target clips show a person following an intended plan, then visibly diverging from it in a way where a streaming assistant could help them recover.

## Quick Start

```powershell
python -m pipeline.run_pipeline --candidates data/candidates.csv --out site/data/dataset.json --report build/report.md
python -m http.server 8000 -d site
```

Then open http://localhost:8000.

## Publish at `/dataset/`

Create a GitHub repository named `dataset`, push this project to it, and enable **Settings > Pages > Source > GitHub Actions**. The included workflow publishes the `site/` folder, so the public URL will be:

```text
https://<your-github-username>.github.io/dataset/
```

## Project Layout

- `data/source_map.json` - seed source families and academic dataset pointers.
- `data/candidates.csv` - editable candidate clip metadata.
- `pipeline/` - cue scoring, normalization, and manifest generation.
- `site/` - static dataset browser that reads `site/data/dataset.json`.
- `build/` - generated reports.

## Candidate CSV Schema

Add rows to `data/candidates.csv` with:

```csv
id,title,description,transcript,url,source_family,domain,goal,timestamp_seconds,license
```

The pipeline scores title, description, and transcript text using deviation cues from the source map. It also creates a compact annotation stub for each candidate so a human or VLM verifier can finish the labels.

## Dataset Framing

The benchmark task is:

> Detect when the user's observed trajectory is diverging from their intended plan, and produce a timely intervention that helps them recover.

The generated manifest contains source cards plus candidate clip records with cue scores, inferred deviation types, intervention windows, and assistant-message drafts.
