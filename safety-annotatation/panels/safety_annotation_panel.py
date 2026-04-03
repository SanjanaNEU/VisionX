"""Panel: Twelve Labs upload, embeddings preview, Pegasus safety."""

from __future__ import annotations

import json
import os
from html import escape as html_escape
from urllib.parse import quote

import fiftyone.operators.types as types
from fiftyone import ViewField as F
from fiftyone.operators.panel import Panel, PanelConfig

from ..twelve_labs_helpers import (
    TWELVELABS_SETUP_MESSAGE,
    analyze_safety,
    fetch_embedding_summary,
    get_client,
    is_twelvelabs_configured,
    upload_video_file,
)

MAX_BULK = 100
NUM_THUMB_SLOTS = 12
THUMB_COLS = 4
FILTER_ALL = "__all__"

VIDEO_EXT = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v", ".mpg", ".mpeg"}
IMAGE_EXT = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


def _media_url(filepath: str) -> str:
    return "/media?filepath=" + quote(filepath, safe="")


def _thumb_overlay_markdown(url: str, is_video: bool, class_label, sample_id: str) -> str:
    """HTML block: media with class + id badge top-left (MarkdownView)."""
    cls = html_escape(str(class_label) if class_label is not None else "—")
    sid = html_escape(str(sample_id))
    u = html_escape(url, quote=True)
    if is_video:
        media = (
            f'<video src="{u}" muted playsInline preload="metadata" '
            'style="width:100%;height:100%;object-fit:cover;display:block;"></video>'
        )
    else:
        media = (
            f'<img src="{u}" alt="" '
            'style="width:100%;height:100%;object-fit:cover;display:block;" />'
        )
    badge = (
        f'<div style="position:absolute;top:0;left:0;text-align:left;max-width:calc(100% - 4px);'
        f'margin:2px;padding:2px 4px;font-size:10px;line-height:1.15;color:#fff;'
        f'background:rgba(0,0,0,0.65);border-radius:2px;word-break:break-word;'
        f'pointer-events:none;">'
        f"<strong>{cls}</strong><br/>"
        f'<span style="opacity:0.9;font-size:9px;">{sid}</span></div>'
    )
    return (
        f'<div style="position:relative;width:96px;height:64px;margin:0;padding:0;'
        f'line-height:0;">{media}{badge}</div>'
    )


def _distinct_gt_labels(dataset):
    try:
        vals = dataset.distinct("ground_truth.label")
    except Exception:
        return []
    out = {v for v in (vals or []) if v is not None}
    return sorted(out, key=lambda x: str(x))


def _base_view(ctx):
    if ctx.view is not None:
        return ctx.view
    return ctx.dataset


def _filtered_view(ctx, filter_labels):
    view = _base_view(ctx)
    if not filter_labels:
        return view
    return view.match(F("ground_truth.label").is_in(filter_labels))


def _parse_bulk_ids_json(state: dict) -> list:
    raw = state.get("bulk_sample_ids", "[]")
    if isinstance(raw, list):
        return [str(x) for x in raw]
    try:
        ids = json.loads(raw) if raw else []
        return [str(x) for x in ids] if isinstance(ids, list) else []
    except Exception:
        return []


def _effective_label_filter(ctx) -> list:
    v = ctx.params.get("filter_label_choice")
    if v is None:
        v = ctx.panel.state.get("filter_label_choice", FILTER_ALL)
    if v in (None, "", FILTER_ALL):
        return []
    return [str(v)]


def _filtered_ids_for_tree(ctx, filter_labels):
    try:
        fv = _filtered_view(ctx, filter_labels)
        return fv.limit(MAX_BULK).values("id")
    except Exception:
        return []


def _thumb_slot_ids(ctx, filter_labels) -> list:
    """Fixed-length list of sample ids (or '') for thumbnail row."""
    ids = _filtered_ids_for_tree(ctx, filter_labels)[:NUM_THUMB_SLOTS]
    out = [str(x) for x in ids]
    while len(out) < NUM_THUMB_SLOTS:
        out.append("")
    return out


def _coerce_bool(v) -> bool:
    if v is True:
        return True
    if v is False or v is None:
        return False
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return bool(v)


def _resolve_sample_id(ctx):
    """Prefer modal current sample, else single selection, else first in view."""
    if ctx.current_sample:
        return ctx.current_sample
    sel = ctx.selected or []
    if len(sel) == 1:
        return sel[0]
    if len(sel) > 1:
        raise ValueError("Select only one sample, or open a sample in the modal.")
    view = ctx.view
    if view is None or len(view) == 0:
        raise ValueError("No samples in the current view.")
    return view.first().id


def _ground_truth_label(sample):
    gt = sample.get("ground_truth", None)
    if gt is None:
        return None
    if hasattr(gt, "label"):
        return gt.label
    if isinstance(gt, dict):
        return gt.get("label")
    return None


class SafetyAnnotationPanel(Panel):
    """Upload to TL index, embedding preview, Pegasus safety JSON."""

    @property
    def config(self):
        return PanelConfig(
            name="safety_annotation_panel",
            label="Safety + Twelve Labs",
            surfaces="grid",
        )

    def on_apply_thumb_selection(self, ctx):
        labels = _effective_label_filter(ctx)
        slot_ids = _thumb_slot_ids(ctx, labels)
        picked = []
        for i in range(NUM_THUMB_SLOTS):
            sid = slot_ids[i] if i < len(slot_ids) else ""
            if not sid:
                continue
            if _coerce_bool(ctx.params.get(f"thumb_sel_{i}")):
                picked.append(sid)
        picked = picked[:MAX_BULK]
        ctx.panel.set_state("bulk_sample_ids", json.dumps(picked))
        msg = f"Upload queue: {len(picked)} sample(s) from checked thumbnails."
        ctx.panel.set_state("last_message", msg)
        ctx.ops.notify(msg, variant="success")
        return {}

    def on_show_filtered_in_app(self, ctx):
        dataset = ctx.dataset
        if dataset is None:
            ctx.ops.notify("No dataset loaded.", variant="warning")
            return {}
        labels = _effective_label_filter(ctx)
        ids = _filtered_ids_for_tree(ctx, labels)
        if not ids:
            ctx.ops.notify("No samples in the filtered view.", variant="warning")
            return {}
        ctx.ops.show_samples(ids, use_extended_selection=False)
        ctx.ops.notify(
            f"Showing {len(ids)} sample(s) in the App (cap {MAX_BULK}).",
            variant="success",
        )
        return {}

    def on_select_queued_in_app(self, ctx):
        ids = _parse_bulk_ids_json(ctx.panel.state)
        if not ids:
            ctx.ops.notify(
                "Upload queue is empty. Check thumbnails or use Queue buttons.",
                variant="warning",
            )
            return {}
        ctx.ops.set_selected_samples(ids)
        ctx.ops.notify(
            f"Selected {len(ids)} sample(s) in the App.",
            variant="success",
        )
        return {}

    def on_use_app_selection(self, ctx):
        dataset = ctx.dataset
        if dataset is None:
            ctx.ops.notify("No dataset loaded.", variant="warning")
            return {}
        labels = _effective_label_filter(ctx)
        try:
            fv = _filtered_view(ctx, labels)
            allowed = set(fv.values("id"))
        except Exception as exc:
            ctx.ops.notify(f"Filter failed: {exc}", variant="warning")
            return {}
        sel = [str(x) for x in (ctx.selected or [])]
        ids = [x for x in sel if x in allowed][:MAX_BULK]
        ctx.panel.set_state("bulk_sample_ids", json.dumps(ids))
        msg = f"Queued {len(ids)} sample(s) from App selection (cap {MAX_BULK})."
        ctx.panel.set_state("last_message", msg)
        ctx.ops.notify(msg, variant="success")
        return {}

    def on_select_all_filtered(self, ctx):
        dataset = ctx.dataset
        if dataset is None:
            ctx.ops.notify("No dataset loaded.", variant="warning")
            return {}
        labels = _effective_label_filter(ctx)
        try:
            fv = _filtered_view(ctx, labels)
            n_total = len(fv)
            ids = fv.limit(MAX_BULK).values("id")
        except Exception as exc:
            ctx.ops.notify(f"Filter failed: {exc}", variant="warning")
            return {}
        ctx.panel.set_state("bulk_sample_ids", json.dumps(ids))
        msg = (
            f"Queued {len(ids)} sample(s) (cap {MAX_BULK}). "
            f"Filtered view has {n_total} total."
        )
        ctx.panel.set_state("last_message", msg)
        ctx.ops.notify(msg, variant="success")
        return {}

    def on_clear_bulk(self, ctx):
        ctx.panel.set_state("bulk_sample_ids", json.dumps([]))
        ctx.panel.set_state("last_message", "Upload queue cleared.")
        ctx.ops.notify("Upload queue cleared.", variant="success")
        return {}

    def on_run_action(self, ctx):
        if not is_twelvelabs_configured(ctx):
            ctx.panel.set_state("last_message", "Twelve Labs API key missing.")
            ctx.ops.notify(TWELVELABS_SETUP_MESSAGE, variant="warning")
            return {}

        action = ctx.params.get("panel_action") or ctx.panel.state.get(
            "panel_action", "upload_current"
        )
        dataset = ctx.dataset
        if dataset is None:
            ctx.ops.notify("No dataset loaded.", variant="warning")
            return {}

        index_id = (dataset.info or {}).get("tl_index_id")
        if not index_id:
            ctx.ops.notify(
                "No index id. Run the operator Create Twelve Labs index first.",
                variant="warning",
            )
            return {}

        bulk_ids = _parse_bulk_ids_json(ctx.panel.state)
        if action == "upload_current" and bulk_ids:
            try:
                client = get_client(ctx)
            except Exception as exc:
                ctx.ops.notify(str(exc), variant="warning")
                return {}
            uploaded = skipped = failed = 0
            for sid in bulk_ids:
                try:
                    sample = dataset[sid]
                except Exception:
                    failed += 1
                    continue
                filepath = sample.filepath
                if not filepath or not os.path.isfile(filepath):
                    skipped += 1
                    continue
                if sample.get("tl_video_id"):
                    skipped += 1
                    continue
                try:
                    video_id = upload_video_file(client, index_id, filepath)
                    sample["tl_video_id"] = video_id
                    sample.save()
                    uploaded += 1
                except Exception:
                    failed += 1
            msg = (
                f"Bulk upload done: uploaded={uploaded}, skipped={skipped}, failed={failed}."
            )
            ctx.panel.set_state("last_message", msg)
            ctx.panel.set_state("embedding_preview", "")
            ctx.panel.set_state("safety_output", "")
            ctx.ops.notify(msg, variant="success" if failed == 0 else "warning")
            return {}

        try:
            sid = _resolve_sample_id(ctx)
            sample = dataset[sid]
        except ValueError as e:
            ctx.ops.notify(str(e), variant="warning")
            return {}
        except Exception as exc:
            ctx.ops.notify(f"Could not load sample: {exc}", variant="warning")
            return {}

        filepath = sample.filepath
        if not filepath or not os.path.isfile(filepath):
            ctx.ops.notify("Sample has no readable filepath.", variant="warning")
            return {}

        try:
            client = get_client(ctx)
        except Exception as exc:
            ctx.ops.notify(str(exc), variant="warning")
            return {}

        try:
            if action == "upload_current":
                video_id = upload_video_file(client, index_id, filepath)
                sample["tl_video_id"] = video_id
                sample.save()
                msg = f"Uploaded. tl_video_id={video_id}"
                ctx.panel.set_state("last_message", msg)
                ctx.panel.set_state("embedding_preview", "")
                ctx.panel.set_state("safety_output", "")
                ctx.ops.notify(msg, variant="success")

            elif action == "view_embeddings":
                vid = sample.get("tl_video_id", None)
                if not vid:
                    ctx.ops.notify(
                        "No tl_video_id on this sample. Upload first.",
                        variant="warning",
                    )
                    return {}
                preview = fetch_embedding_summary(client, index_id, vid)
                sample["tl_embedding_preview"] = preview
                sample.save()
                ctx.panel.set_state("embedding_preview", preview)
                ctx.panel.set_state("last_message", "Retrieved embedding preview.")
                ctx.ops.notify("Embedding preview updated.", variant="success")

            elif action == "pegasus_safety":
                vid = sample.get("tl_video_id", None)
                if not vid:
                    ctx.ops.notify(
                        "No tl_video_id on this sample. Upload first.",
                        variant="warning",
                    )
                    return {}
                label = _ground_truth_label(sample)
                result = analyze_safety(client, vid, ground_truth_label=label)
                reasoning = result.get("reasoning", result.get("raw", str(result)))
                score = result.get("danger_score", "")
                text_out = f"reasoning: {reasoning}\ndanger_score: {score}"
                sample["tl_safety_reasoning"] = str(reasoning)
                if score != "":
                    sample["tl_danger_score"] = score
                sample.save()
                ctx.panel.set_state("safety_output", text_out)
                ctx.panel.set_state("last_message", "Pegasus analysis complete.")
                ctx.ops.notify("Safety analysis complete.", variant="success")
            else:
                ctx.ops.notify(f"Unknown action: {action}", variant="warning")
        except Exception as exc:
            ctx.ops.notify(f"Action failed: {exc}", variant="warning")

    def _build_thumb_cell(self, cell, i, sid, dataset, bulk_set, gt_label):
        fp = None
        if dataset is not None and sid:
            try:
                smp = dataset[sid]
                fp = smp.filepath
            except Exception:
                fp = None

        if fp and os.path.isfile(fp):
            url = _media_url(fp)
            ext = os.path.splitext(fp)[1].lower()
            is_video = ext in VIDEO_EXT if ext else False
            if ext in IMAGE_EXT:
                is_video = False
            md = _thumb_overlay_markdown(url, is_video, gt_label, sid)
            cell.md(md, name=f"thumb_block_{i}")
            cell.bool(
                f"thumb_sel_{i}",
                label="Queue",
                view=types.CheckboxView(),
                default=(sid in bulk_set),
            )
        else:
            cell.view(
                f"ph_{i}",
                types.HeaderView(
                    label="—" if not sid else "No file",
                ),
            )
            cell.bool(
                f"thumb_sel_{i}",
                label="Queue",
                view=types.CheckboxView(),
                default=False,
            )

    def _build_thumb_grid(self, left, ctx, dataset, st):
        labels = _effective_label_filter(ctx)
        slot_ids = _thumb_slot_ids(ctx, labels) if dataset is not None else [""] * NUM_THUMB_SLOTS
        bulk_set = set(_parse_bulk_ids_json(st))

        # Single GridView: nested h_stack rows render as "Unsupported view" in panels.
        thumb_grid = types.Object()
        left.define_property(
            "thumb_grid",
            thumb_grid,
            view=types.GridView(
                orientation="2d",
                gap=0,
                align_x="center",
                componentsProps={
                    "sx": {
                        "display": "grid",
                        "gridTemplateColumns": f"repeat({THUMB_COLS}, max-content)",
                        "gridAutoRows": "max-content",
                        "width": "100%",
                        "justifyContent": "start",
                        "columnGap": "2px",
                        "rowGap": "2px",
                    }
                },
            ),
        )
        for i in range(NUM_THUMB_SLOTS):
            cell = types.Object()
            thumb_grid.define_property(
                f"thumb_cell_{i}",
                cell,
                view=types.VStackView(
                    orientation="vertical",
                    gap=0,
                    componentsProps={
                        "sx": {
                            "p": 0,
                            "m": 0,
                            "minWidth": 0,
                            "width": "max-content",
                            "alignItems": "center",
                        }
                    },
                ),
            )
            sid = slot_ids[i] if i < len(slot_ids) else ""
            gt_label = None
            if dataset is not None and sid:
                try:
                    gt_label = _ground_truth_label(dataset[sid])
                except Exception:
                    gt_label = None
            self._build_thumb_cell(cell, i, sid, dataset, bulk_set, gt_label)

    def render(self, ctx):
        dataset = ctx.dataset
        panel = types.Object()
        main_row = panel.h_stack("main_row", gap=2)
        left = main_row.v_stack("left_col", space=5)
        right = main_row.v_stack("right_col", space=7)

        st = ctx.panel.state
        filter_default = st.get("filter_label_choice", FILTER_ALL)

        # --- Left: class dropdown + selectable thumbnails (4 per row) ---
        if dataset is None:
            left.view(
                "no_ds",
                types.Notice(
                    label="Samples",
                    description="Load a dataset to filter by class and show thumbnails.",
                ),
            )
        else:
            distinct = _distinct_gt_labels(dataset)
            dd_class = types.Dropdown()
            dd_class.add_choice(
                FILTER_ALL,
                label="All labels (current App view)",
            )
            for lab in distinct:
                dd_class.add_choice(str(lab), label=str(lab))
            class_values = dd_class.values()
            class_default = (
                filter_default
                if filter_default in class_values
                else FILTER_ALL
            )
            left.enum(
                "filter_label_choice",
                class_values,
                view=dd_class,
                label="Class filter",
                description=(
                    "Restrict thumbnails to `ground_truth.label` within the current view. "
                    "Choose **All labels** to use the full current view."
                ),
                default=class_default,
            )

        left.view(
            "thumbs_header",
            types.Notice(
                label="Thumbnails",
                description=(
                    f"Up to {NUM_THUMB_SLOTS} samples ({THUMB_COLS} per row). "
                    "Check **Queue**, then **Apply checked to upload queue**."
                ),
            ),
        )

        if dataset is not None:
            self._build_thumb_grid(left, ctx, dataset, st)

        left.btn(
            "btn_apply_thumbs",
            label="Apply checked to upload queue",
            variant="contained",
            on_click=self.method_to_uri("on_apply_thumb_selection"),
        )
        left.btn(
            "btn_show_filtered",
            label="Show filtered in App",
            variant="outlined",
            on_click=self.method_to_uri("on_show_filtered_in_app"),
        )
        left.btn(
            "btn_select_queued_app",
            label="Select queue in App",
            variant="outlined",
            on_click=self.method_to_uri("on_select_queued_in_app"),
        )
        left.btn(
            "btn_select_all",
            label="Queue all in filter",
            variant="outlined",
            on_click=self.method_to_uri("on_select_all_filtered"),
        )
        left.btn(
            "btn_use_selection",
            label="Queue App selection",
            variant="outlined",
            on_click=self.method_to_uri("on_use_app_selection"),
        )
        left.btn(
            "btn_clear_bulk",
            label="Clear queue",
            variant="outlined",
            on_click=self.method_to_uri("on_clear_bulk"),
        )

        # --- Right: Twelve Labs + actions ---
        if not is_twelvelabs_configured(ctx):
            right.view(
                "tl_fallback",
                types.Warning(
                    label="Twelve Labs not configured",
                    description=TWELVELABS_SETUP_MESSAGE,
                ),
            )
        else:
            right.view(
                "tl_ready",
                types.Notice(
                    label="Twelve Labs",
                    description="API key detected. Create an index with the operator if needed.",
                ),
            )

        idx = (dataset.info or {}).get("tl_index_id") if dataset else None
        desc = (
            f"tl_index_id: {idx}"
            if idx
            else "No index yet — run operator Create Twelve Labs index."
        )
        right.view(
            "idx_notice", types.Notice(label="Twelve Labs index", description=desc)
        )

        dd = types.Dropdown()
        dd.add_choice(
            "upload_current",
            label="Upload current sample to index",
        )
        dd.add_choice(
            "view_embeddings",
            label="View embeddings (preview)",
        )
        dd.add_choice(
            "pegasus_safety",
            label="Pegasus safety analysis (1–10)",
        )

        right.enum(
            "panel_action",
            dd.values(),
            view=dd,
            label="Action",
            default=st.get("panel_action", "upload_current"),
        )
        right.str(
            "last_message",
            label="Status",
            view=types.ReadOnlyView(),
            default=st.get("last_message", ""),
        )
        right.str(
            "embedding_preview",
            label="Embedding preview",
            view=types.ReadOnlyView(),
            default=st.get("embedding_preview", ""),
        )
        right.str(
            "safety_output",
            label="Pegasus output",
            view=types.ReadOnlyView(),
            default=st.get("safety_output", ""),
        )
        right.btn(
            "run_btn",
            label="Run",
            variant="contained",
            on_click=self.method_to_uri("on_run_action"),
        )
        return types.Property(panel)
