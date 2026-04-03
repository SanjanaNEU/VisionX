# Safety Annotation — FiftyOne plugin

A [FiftyOne](https://docs.voxel51.com/) plugin that connects video samples in your dataset to [Twelve Labs](https://www.twelvelabs.io/): create a multimodal index (Marengo 3.0 + Pegasus 1.2), upload clips, preview Marengo embeddings, and run Pegasus for structured **workplace / industrial safety** analysis with a 1–10 danger score returned as JSON.

The plugin is defined by `fiftyone.yml` and registered from `__init__.py`.

---

## Requirements

- **FiftyOne** ≥ 0.24 (see `fiftyone.yml`)
- **Python** 3.11+ recommended (match your FiftyOne environment)
- **`twelvelabs`** Python SDK (listed in the repo `requirements.txt`)
- **Twelve Labs API key** with access to indexing and analyze APIs
- Video samples must have a valid **`filepath`** on disk (local files the SDK can open)

---

## Install the plugin

FiftyOne loads plugins from **`plugins_dir`** (default under your FiftyOne dataset directory, e.g. `~/fiftyone/__plugins__`). You can print the exact path:

```bash
python -c "import fiftyone as fo; print(fo.config.plugins_dir)"
```

### Local development (symlink)

Point a name under `plugins_dir` at this folder (manifest name is **`safety-annotation`**):

```bash
REPO="/absolute/path/to/VisionX/safety-annotatation"
DEST="$(python -c "import fiftyone as fo; print(fo.config.plugins_dir)")/safety-annotation"
ln -sfn "$REPO" "$DEST"
```

Use `safety-annotatation` in the path if that is how the directory is named in your clone.

### Download from GitHub

When the repo is online, download the plugin subtree that contains `fiftyone.yml`:

```bash
fiftyone plugins download "https://github.com/<user>/VisionX/tree/<branch>/safety-annotatation" \
  --plugin-names safety-annotation
```

Restart the FiftyOne App after adding, moving, or upgrading a plugin.

---

## Configure the API key

The plugin reads **`TWELVELABS_API_KEY`**. Supported sources:

1. **FiftyOne App** — Settings → Plugin secrets, add secret name `TWELVELABS_API_KEY`.
2. **Environment** — `export TWELVELABS_API_KEY=...`
3. **Alternate env name** — `TWELVE_LABS_API_KEY` is also accepted (see `twelve_labs_helpers.py`).

The key is declared under `secrets` in `fiftyone.yml` so the App can prompt for it.

---

## Quick workflow

1. Load a FiftyOne dataset whose samples are **video** with `filepath` set.
2. Optional but recommended: ensure samples have **`ground_truth.label`** — Pegasus will receive it as context when analyzing safety.
3. In the App, run the operator **Create Twelve Labs index** and give the index a name. This creates a Twelve Labs index with **marengo3.0** and **pegasus1.2** (visual + audio) and stores the id in **`dataset.info["tl_index_id"]`**.
4. Open the panel **Safety + Twelve Labs** (grid surface).
5. Select samples in the App grid, choose an action from the dropdown, and click **Run**.

---

## Operator: `create_twelvelabs_index`

| | |
| --- | --- |
| **Label** | Create Twelve Labs index |
| **Purpose** | Creates a new Twelve Labs index and saves its id on the dataset. |
| **Input** | Index name (required when the API key is set). |
| **Side effects** | Sets `dataset.info["tl_index_id"]` and saves the dataset. |

If the API key is missing, the form shows a warning with setup instructions; execution is a no-op with a notification.

Delegated execution is allowed (`allow_delegated_execution=True`).

---

## Panel: `safety_annotation_panel`

| | |
| --- | --- |
| **Label** | Safety + Twelve Labs |
| **Surface** | Grid |

### Layout

The panel shows:

- **Status notice** — whether the Twelve Labs API key is detected and which index id is active.
- **Action dropdown** — choose what to do when **Run** is clicked.
- **Status** (read-only text) — last operation result or progress message.
- **Pegasus output (JSON)** (read-only multiline textbox) — full JSON output from the most recent Pegasus run.
- **Run button** — executes the selected action.

### Actions

| Action | Scope | What it does |
| --- | --- | --- |
| **Upload all selected samples** | All selected in App grid | Uploads each selected sample's video file to the dataset's `tl_index_id`, waits for indexing, and saves **`tl_video_id`** on each sample. Skips samples with no readable `filepath`. Prints a per-item progress count in the Status field. |
| **Upload current / first selected sample** | Single sample | Uploads the currently open (modal) sample, or the first selected sample if no modal is open. Saves **`tl_video_id`** on success. |
| **View embeddings (preview)** | Single sample | Retrieves Marengo visual embedding metadata for the sample's `tl_video_id` and stores a short summary string (`segments`, `dim`, first 8 floats) in **`tl_embedding_preview`**. |
| **Pegasus safety analysis – current sample (JSON)** | Single sample | Runs Pegasus on the current/first selected sample. Returns structured JSON with `reasoning` (string) and `danger_score` (integer 1–10). Saves **`tl_safety_reasoning`** and **`tl_danger_score`** on the sample. Full JSON is displayed in the **Pegasus output** textbox. |
| **Pegasus safety analysis – all selected samples (JSON)** | All selected in App grid | Runs Pegasus on every selected sample that has a `tl_video_id`. Results for all samples are collected into a JSON array and displayed in the **Pegasus output** textbox. Each entry includes `sample_id` plus the Pegasus fields, or an `error` key if that sample failed. Saves `tl_safety_reasoning` and `tl_danger_score` on each successfully analyzed sample. |

**Single-sample resolution order:** modal current sample → first id in `ctx.selected` → first sample in current view.

### Pegasus output format

Single-sample run:

```json
{
  "sample_id": "abc123",
  "reasoning": "The worker is not wearing PPE in the early portion of the clip...",
  "danger_score": 8
}
```

All-selected run:

```json
[
  {
    "sample_id": "abc123",
    "reasoning": "Forklift operating without a spotter...",
    "danger_score": 7
  },
  {
    "sample_id": "def456",
    "error": "no tl_video_id – upload first"
  }
]
```

---

## Fields written by the plugin

| Location | Field | Meaning |
| --- | --- | --- |
| Dataset | `info["tl_index_id"]` | Twelve Labs index id after running the create-index operator. |
| Sample | `tl_video_id` | Twelve Labs video id after a successful upload. |
| Sample | `tl_embedding_preview` | Text summary from Marengo embedding retrieval. |
| Sample | `tl_safety_reasoning` | Pegasus reasoning text. |
| Sample | `tl_danger_score` | Pegasus danger score (1–10) when returned as an integer. |

---

## Implementation notes

- **Index models** — `create_index()` in `twelve_labs_helpers.py` uses **marengo3.0** and **pegasus1.2** with options `["visual", "audio"]`.
- **Upload** — Uses the Twelve Labs tasks API, waits until the task completes (`tasks.wait_for_done`), then reads `video_id` from the completed task.
- **Analyze** — `analyze_safety()` uses `ResponseFormat` with a JSON schema (`reasoning`, `danger_score`); malformed JSON falls back to storing raw text.
- **Ground truth context** — When a sample has `ground_truth.label`, it is prepended to the Pegasus prompt so the model knows the dataset label for this clip.

---

## Troubleshooting

- **"Twelve Labs API key missing"** — Set `TWELVELABS_API_KEY` in App plugin secrets or the environment; restart the App if needed.
- **"No index id"** — Run **Create Twelve Labs index** on the loaded dataset first.
- **"No tl_video_id"** — Upload the sample (or bulk-upload all selected) before running embeddings or Pegasus.
- **Upload skipped** — Common causes: file missing on disk, or `tl_video_id` already set on that sample.
- **Pegasus output shows `"error"` entries** — Those samples were either not loaded or are missing `tl_video_id`; upload them first.

---

## Manifest and package layout

| File | Role |
| --- | --- |
| `fiftyone.yml` | Plugin name `safety-annotation`, version, FiftyOne version, operators/panels list, secrets. |
| `__init__.py` | `register(pctx)` — registers `CreateTwelveLabsIndex` and `SafetyAnnotationPanel`. |
| `twelve_labs_helpers.py` | API key resolution, client, index create, upload, embedding preview, Pegasus safety analyze. |
| `operators/create_twelvelabs_index.py` | Create-index operator. |
| `panels/safety_annotation_panel.py` | Panel UI and action handlers. |

---

## License

See `fiftyone.yml` (Apache 2.0 as declared in the manifest).

---

## Author

Sathwik Matcha (see `fiftyone.yml`).
