# VisionX

# RedFlag 🪦

> *It doesn't prevent defects. It just shows up after and tells you exactly when the patient flatlined.*

**RedFlag** is an AI quality supervisor that watches your assembly line footage and automatically timestamps every moment something looks wrong — with zero labels, zero training, and absolutely zero sympathy for your scrap rate.

Feed it a shift's worth of video. It hands you back a timestamped defect report, a plain-English cause-of-death for each anomaly, and a 1–10 severity score. You go from "something went wrong on line 3" to root cause in under a minute.

Your human QA team can keep doing whatever they were doing. RedFlag watched the whole shift. It always does.

---

## What It Does

```
Assembly line footage  →  RedFlag  →  Timestamped defect log
                                          →  "What went wrong" in plain English
                                          →  Severity score 1–10
                                          →  Root cause hypothesis
                                          →  Visual anomaly timeline in FiftyOne App
```

No model training. No labeled defect data. No PhD required.

It uses embedding-based anomaly detection: everything that looks like "normal production" gets compressed into a baseline. Anything that drifts from that baseline gets flagged, described, and scored. The line teaches RedFlag what healthy looks like. RedFlag teaches you what sick looks like.

---

## The Problem It Solves

Traditional quality control on assembly lines requires one of three things:

1. A very attentive human watching hours of footage (expensive, fallible, hates their job)
2. A trained CV model with thousands of labeled defect examples (slow to build, brittle to new failure modes)
3. Physical sensors wired into the machine (infrastructure project, not a Tuesday afternoon)

RedFlag requires none of these. It requires video and the grim acceptance that things will break.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Video dataset management | [FiftyOne](https://docs.voxel51.com/) |
| Anomaly detection (embeddings) | [Twelve Labs Marengo 3.0](https://www.twelvelabs.io/blog/marengo-3-0) |
| Defect description + scoring | [Twelve Labs Pegasus 1.2](https://www.twelvelabs.io/blog/introducing-pegasus-1-2) |
| Visualization | FiftyOne App + Plotly |
| Language | Python 3.11+ |

**Marengo 3.0** generates 512-dimensional video embeddings. RedFlag builds a baseline distribution from your normal-operation footage, then flags every clip whose embedding drifts too far from it. No labels. No training. Just distance from normal.

**Pegasus 1.2** watches the flagged clips and tells you what it sees: component misalignment, machine jam, wrong assembly sequence, missing part, operator error. It gives you timestamps for the exact moment things went sideways and a severity score from 1 (eyebrow raise) to 10 (stop the line immediately).

---

## Installation

```bash
# Prerequisites: Python 3.11+, FFmpeg
pip install fiftyone twelvelabs numpy scikit-learn plotly python-dotenv

# Install GlitchCoroner as a FiftyOne plugin
fiftyone plugins download https://github.com/YOUR_USERNAME/glitchcoroner

# Set your Twelve Labs API key
export TWELVELABS_API_KEY="your_key_here"
# (or copy .env.example to .env and fill it in)
```

Get a free Twelve Labs API key at [playground.twelvelabs.io](https://playground.twelvelabs.io) — 600 minutes of indexing included.

---

## Quickstart

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

# Load the Safe & Unsafe Behaviours dataset (or your own footage)
dataset = load_from_hub("Voxel51/Safe-and-Unsafe-Behaviours")

# Launch the FiftyOne App
session = fo.launch_app(dataset)
```

Then in the FiftyOne App:

1. Open the **RedFlag** panel
2. Click **Run Autopsy** (`analyze_line` operator)
3. Watch it timestamp every moment something went wrong
4. Click any flagged clip → see the defect description + severity score + exact timestamp

---

## FiftyOne Plugin Operators

| Operator | What It Does |
|---|---|
| `analyze_line` | Embeds all footage, scores anomalies, calls Pegasus on flagged clips |
| `set_normal_baseline` | Define what "healthy" looks like using selected clips |
| `filter_by_severity` | Show only clips above a minimum severity threshold |

---

## Output Fields Written Per Clip

| Field | Type | Example |
|---|---|---|
| `gc_anomaly_score` | float | `0.847` |
| `gc_is_anomaly` | bool | `True` |
| `gc_severity_score` | int 1–10 | `8` |
| `gc_defect_description` | str | `"Component installed at wrong angle at 00:14"` |
| `gc_defect_type` | str | `"misalignment"` |
| `gc_root_cause_hypothesis` | str | `"Feed mechanism hesitation before insertion"` |
| `gc_peak_timestamp` | str | `"00:14"` |
| `gc_confidence` | str | `"high"` |

---

## Dashboard Panel

The GlitchCoroner panel in the FiftyOne App shows:

- **Severity Timeline** — a scrollable plot of severity scores across the shift
- **Defect Type Breakdown** — what categories of problems are most common
- **Top Offenders** — the 10 worst clips ranked by severity
- **Embedding Scatter** — 2D projection of all clips; anomalies cluster visibly away from the normal blob
- **Per-clip Autopsy Report** — rendered inline next to the video player

---

## How Anomaly Detection Works

```
1. Index all clips with Marengo 3.0  →  512-d embedding per clip
2. Build normal baseline             →  centroid of "healthy" clip embeddings
3. Score every clip                  →  cosine distance from baseline centroid
4. Compute z-scores                  →  standardize across the shift
5. Flag outliers                     →  clips beyond threshold (default: 2.0σ)
6. Call Pegasus on flagged clips     →  get description, type, score, timestamp
7. Write everything to FiftyOne      →  browse in App, export for reports
```

Threshold is configurable. Lower it to catch subtle drift. Raise it if your line is noisy and you only want to see disasters.

---

## Adding Your Own Footage

```bash
# Add scraped or recorded footage
python scripts/ingest_footage.py \
  --input /path/to/your/shift_footage/ \
  --source "line_3_morning_shift"

# Or use the built-in scraper for training/test data
python scripts/scrape_videos.py \
  --keywords "assembly line defect" "manufacturing quality failure" \
  --max-clips 50
```

GlitchCoroner will:
- Validate codec, duration, resolution via FFmpeg
- Merge clips into the FiftyOne dataset
- Tag them with their source for traceability

---

## Project Structure

```
glitchcoroner/
├── fiftyone.yml           # Plugin manifest
├── __init__.py            # Operators + panel
├── glitchcoroner/
│   ├── anomaly_detector.py   # Marengo embedding + scoring
│   ├── defect_analyzer.py    # Pegasus defect analysis + parsing
│   ├── dataset_loader.py     # FiftyOne loading utilities
│   └── constants.py          # Prompts, field names, thresholds
├── scripts/
│   ├── ingest_footage.py     # Import your own video
│   ├── scrape_videos.py      # yt-dlp scraper
│   └── run_eda.py            # EDA stats
└── docs/
    ├── PRD.md
    ├── TRD.md
    └── WORKFLOW.md
```

---

## Dataset

Primary dataset: **Safe & Unsafe Behaviours** (`Voxel51/Safe-and-Unsafe-Behaviours`)
691 clips from a Turkish manufacturing facility, 1080p, 24 FPS, 8 behavior classes.

RedFlag is not limited to this dataset. Point it at any video footage of a process that has a "normal" state. Assembly lines, quality inspection stations, conveyor belts, packaging lines, CNC machines. If it moves and sometimes breaks, GlitchCoroner will notice.

---

## Requirements

- Python 3.11+
- FFmpeg (for video validation)
- Twelve Labs API key ([free tier: 600 minutes](https://playground.twelvelabs.io))
- FiftyOne ≥ 0.24
- Enough disk space for your footage

---

## Limitations

- **Batch only (v0.1):** GlitchCoroner processes recorded footage. It does not stream live video yet.
- **Pegasus can hallucinate:** Defect descriptions are AI-generated. Treat them as hypotheses, not verdicts. Severity scores are clamped 1–10 but not guaranteed calibrated.
- **Baseline quality matters:** If your "normal" clips include defects, the baseline will be wrong and GlitchCoroner will miss things. Garbage in, confident garbage out.
- **Not a replacement for engineers:** GlitchCoroner tells you *when* and *what*. It does not fix your tooling, retrain your operators, or file your incident reports. That's still your problem.

---

## License

Apache 2.0. Use it, fork it, deploy it, blame it.

---

## Built At

**Video Understanding AI Hackathon @ Northeastern University**
April 3, 2026 · Powered by [FiftyOne](https://voxel51.com) × [Twelve Labs](https://twelvelabs.io)

*"The line never lies. GlitchCoroner just makes it talk."*
