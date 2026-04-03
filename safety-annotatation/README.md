# Safety Annotation — FiftyOne plugin

A [FiftyOne](https://docs.voxel51.com/) plugin that connects video samples in your dataset to [Twelve Labs](https://www.twelvelabs.io/): create a multimodal index (Marengo 3.0 + Pegasus 1.2), upload clips, preview Marengo embeddings, and run Pegasus for structured **workplace / industrial safety** analysis with a 1–10 danger score.

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

If the repository exposes only this plugin at its root, omit `--plugin-names` or adjust to match the `name` field in `fiftyone.yml`.

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
2. Optional but recommended: ensure samples have **`ground_truth.label`** if you want label-based filtering in the panel and Pegasus to see the dataset label as context.
3. In the App, run the operator **Create Twelve Labs index** and give the index a name. This creates a Twelve Labs index with **marengo3.0** and **pegasus1.2** (visual + audio) and stores the id in **`dataset.info["tl_index_id"]`**.
4. Open the panel **Safety + Twelve Labs** (grid surface).
5. For a **single sample**: select one sample or open the modal on one clip, choose an action, click **Run**.
6. For **bulk upload**: set optional comma-separated **Filter labels** (`ground_truth.label`), then **Select all in filter** or **Use App selection** (intersected with the filtered view, max **100** samples). Choose **Upload current sample to index** and **Run** to upload all queued samples that have files and no `tl_video_id` yet.

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

### Label filter

- **Filter labels (comma-separated)** — Restricts bulk helpers and counts to samples whose `ground_truth.label` is in that list, still scoped to the **current App view** (`ctx.view` when present, else full dataset).
- The panel lists distinct `ground_truth.label` values when the field exists.

### Bulk queue (upload)

| Control | Behavior |
| --- | --- |
| **Select all in filter** | Queues up to **100** sample ids from the filtered view. |
| **Use App selection** | Queues the intersection of App selection with the filtered view (cap **100**). |
| **Clear bulk queue** | Empties the queue. |

When the action is **Upload current sample to index** and the bulk queue is non-empty, **Run** performs **bulk upload** (not the single-sample path).

### Actions (dropdown + Run)

| Action | What it does |
| --- | --- |
| **Upload current sample to index** | Uploads the video file to the dataset’s `tl_index_id`, waits until indexing completes, sets **`tl_video_id`** on the sample. Skips samples that already have `tl_video_id` or missing/invalid `filepath`. |
| **View embeddings (preview)** | Reads Marengo visual embedding metadata for the sample’s `tl_video_id` and stores a short summary string on the sample and in the panel (not full vectors). |
| **Pegasus safety analysis (1–10)** | Calls Pegasus with a JSON schema: `reasoning` (string) and `danger_score` (integer 1–10), focused on **industrial / workplace safety**. If `ground_truth.label` exists, it is passed as context. Writes **`tl_safety_reasoning`** and **`tl_danger_score`** on the sample. |

**Single-sample resolution order:** modal current sample → if exactly one id selected → otherwise first sample in the current view. Multiple selection without the modal shows a warning.

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
- **Upload** — Uses the Twelve Labs tasks API, waits until the task completes, then reads `video_id` from the completed task.
- **Analyze** — `analyze_safety()` uses `ResponseFormat` JSON schema; malformed JSON falls back to storing raw text in the parsed structure.

---

## Troubleshooting

- **“Twelve Labs API key missing”** — Set `TWELVELABS_API_KEY` in App plugin secrets or the environment; restart the App if needed.
- **“No index id”** — Run **Create Twelve Labs index** on the loaded dataset first.
- **“No tl_video_id”** — Upload the sample (or bulk-upload) before embeddings or Pegasus actions.
- **Bulk upload skipped** — Common causes: file missing on disk, or `tl_video_id` already set.

---

## Manifest and package layout

| File | Role |
| --- | --- |
| `fiftyone.yml` | Plugin name `safety-annotation`, version, FiftyOne version, operators/panels list, secrets. |
| `__init__.py` | `register(pctx)` — registers `CreateTwelveLabsIndex` and `SafetyAnnotationPanel`. |
| `twelve_labs_helpers.py` | API key resolution, client, index create, upload, embedding preview, Pegasus safety analyze. |
| `operators/create_twelvelabs_index.py` | Create-index operator. |
| `panels/safety_annotation_panel.py` | Panel UI and handlers. |

---

## License

See `fiftyone.yml` (Apache 2.0 as declared in the manifest).

---

## Author

Sathwik Matcha (see `fiftyone.yml`).
