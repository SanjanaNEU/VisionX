"""Panel: Twelve Labs upload, embeddings preview, Pegasus safety."""

from __future__ import annotations

import json
import os

import fiftyone.operators.types as types
from fiftyone.operators.panel import Panel, PanelConfig

from ..twelve_labs_helpers import (
    TWELVELABS_SETUP_MESSAGE,
    analyze_safety,
    fetch_embedding_summary,
    get_client,
    is_twelvelabs_configured,
    upload_video_file,
)


def _normalize_sample_id(raw):
    """Turn App/JSON selection entries into a string sample id."""
    if raw is None:
        return ""
    if isinstance(raw, dict):
        for k in ("id", "_id", "sample_id"):
            if k in raw:
                return _normalize_sample_id(raw[k])
        return ""
    return str(raw).strip()


def _load_sample_for_panel(ctx, sid: str):
    """
    Load a sample for display and Run.

    Prefer ``ctx.view`` (matches what the App shows: filters, patches, clips),
    then fall back to the full ``ctx.dataset`` (e.g. extended selection).
    """
    s = _normalize_sample_id(sid)
    if not s:
        return None, "empty id"
    errors = []
    view = ctx.view
    if view is not None:
        try:
            return view[s], None
        except Exception as e:
            errors.append(f"view: {type(e).__name__}: {e}")
    ds = ctx.dataset
    if ds is not None:
        try:
            return ds[s], None
        except Exception as e:
            errors.append(f"dataset: {type(e).__name__}: {e}")
    return None, "; ".join(errors) if errors else "no dataset/view"


def _resolve_sample_id(ctx):
    """
    Resolve which sample Run targets:

    1. ``ctx.current_sample`` if the sample modal is open.
    2. Else the **first** ID in ``ctx.selected`` (App grid selection).
    3. Else the first sample in the current view.
    """
    if ctx.current_sample:
        return _normalize_sample_id(ctx.current_sample)
    sel = ctx.selected or []
    if sel:
        nid = _normalize_sample_id(sel[0])
        if nid:
            return nid
    view = ctx.view
    if view is None or len(view) == 0:
        raise ValueError("No samples in the current view. Select a sample in the App.")
    return _normalize_sample_id(view.first().id)


def _optional_sample_field(sample, field_name):
    """Read an optional field from a Sample / SampleView (not dict ``.get``)."""
    if sample is None:
        return None
    try:
        return sample[field_name]
    except KeyError:
        return None


def _ground_truth_label(sample):
    """Read ``ground_truth.label`` from a Sample / SampleView."""
    gt = _optional_sample_field(sample, "ground_truth")
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
                "No index id. Run the operator Twelve Labs index first.",
                variant="warning",
            )
            return {}

        try:
            client = get_client(ctx)
        except Exception as exc:
            ctx.ops.notify(str(exc), variant="warning")
            return {}

        # ── bulk upload ──────────────────────────────────────────────────────
        if action == "upload_selected":
            sel_ids = [
                _normalize_sample_id(x)
                for x in (ctx.selected or [])
                if _normalize_sample_id(x)
            ]
            if not sel_ids:
                ctx.ops.notify(
                    "No samples selected in the App. Select at least one.",
                    variant="warning",
                )
                return {}
            ok, skipped, failed = 0, 0, []
            ctx.panel.set_state(
                "last_message",
                f"Uploading 0 / {len(sel_ids)} …",
            )
            for i, sid in enumerate(sel_ids, 1):
                sample, load_err = _load_sample_for_panel(ctx, sid)
                if sample is None:
                    failed.append(f"{sid}: {load_err}")
                    continue
                fp = sample.filepath
                if not fp or not os.path.isfile(fp):
                    skipped += 1
                    continue
                try:
                    video_id = upload_video_file(client, index_id, fp)
                    sample["tl_video_id"] = video_id
                    sample.save()
                    ok += 1
                except Exception as exc:
                    failed.append(f"{sid}: {exc}")
                ctx.panel.set_state(
                    "last_message",
                    f"Uploading {i} / {len(sel_ids)} … ({ok} done)",
                )
            parts = [f"Uploaded {ok} / {len(sel_ids)} samples."]
            if skipped:
                parts.append(f"{skipped} skipped (no file).")
            if failed:
                parts.append(f"{len(failed)} failed: " + "; ".join(failed[:3]))
            msg = " ".join(parts)
            ctx.panel.set_state("last_message", msg)
            ctx.panel.set_state("safety_output", "")
            variant = "success" if not failed else "warning"
            ctx.ops.notify(msg, variant=variant)
            return {}

        # ── pegasus on all selected samples ──────────────────────────────────
        if action == "pegasus_safety_selected":
            sel_ids = [
                _normalize_sample_id(x)
                for x in (ctx.selected or [])
                if _normalize_sample_id(x)
            ]
            if not sel_ids:
                ctx.ops.notify(
                    "No samples selected in the App. Select at least one.",
                    variant="warning",
                )
                return {}
            results = []
            ctx.panel.set_state(
                "last_message", f"Running Pegasus on 0 / {len(sel_ids)} …"
            )
            for i, sid in enumerate(sel_ids, 1):
                sample, load_err = _load_sample_for_panel(ctx, sid)
                if sample is None:
                    results.append({"sample_id": sid, "error": load_err})
                    continue
                vid = _optional_sample_field(sample, "tl_video_id")
                if not vid:
                    results.append(
                        {"sample_id": sid, "error": "no tl_video_id – upload first"}
                    )
                    continue
                label = _ground_truth_label(sample)
                try:
                    result = analyze_safety(client, vid, ground_truth_label=label)
                    reasoning = result.get("reasoning", result.get("raw", str(result)))
                    score = result.get("danger_score", "")
                    sample["tl_safety_reasoning"] = str(reasoning)
                    if score != "":
                        sample["tl_danger_score"] = score
                    sample.save()
                    results.append({"sample_id": sid, **result})
                except Exception as exc:
                    results.append({"sample_id": sid, "error": str(exc)})
                ctx.panel.set_state(
                    "last_message",
                    f"Running Pegasus on {i} / {len(sel_ids)} …",
                )
            json_out = json.dumps(results, indent=2)
            ctx.panel.set_state("safety_output", json_out)
            msg = f"Pegasus analysis complete for {len(sel_ids)} sample(s)."
            ctx.panel.set_state("last_message", msg)
            ctx.ops.notify(msg, variant="success")
            return {}

        # ── single-sample actions ────────────────────────────────────────────
        try:
            sid = _resolve_sample_id(ctx)
        except ValueError as e:
            ctx.ops.notify(str(e), variant="warning")
            return {}

        sample, load_err = _load_sample_for_panel(ctx, sid)
        if sample is None:
            ctx.ops.notify(
                load_err or f"Could not load sample {sid!r}.",
                variant="warning",
            )
            return {}

        filepath = sample.filepath
        if not filepath or not os.path.isfile(filepath):
            ctx.ops.notify("Sample has no readable filepath.", variant="warning")
            return {}

        try:
            if action == "upload_current":
                video_id = upload_video_file(client, index_id, filepath)
                sample["tl_video_id"] = video_id
                sample.save()
                msg = f"Uploaded. tl_video_id={video_id}"
                ctx.panel.set_state("last_message", msg)
                ctx.panel.set_state("safety_output", "")
                ctx.ops.notify(msg, variant="success")

            elif action == "view_embeddings":
                vid = _optional_sample_field(sample, "tl_video_id")
                if not vid:
                    ctx.ops.notify(
                        "No tl_video_id on this sample. Upload first.",
                        variant="warning",
                    )
                    return {}
                preview = fetch_embedding_summary(client, index_id, vid)
                sample["tl_embedding_preview"] = preview
                sample.save()
                ctx.panel.set_state("last_message", "Retrieved embedding preview.")
                ctx.ops.notify("Embedding preview updated.", variant="success")

            elif action == "pegasus_safety":
                vid = _optional_sample_field(sample, "tl_video_id")
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
                sample["tl_safety_reasoning"] = str(reasoning)
                if score != "":
                    sample["tl_danger_score"] = score
                sample.save()
                json_out = json.dumps({"sample_id": sid, **result}, indent=2)
                ctx.panel.set_state("safety_output", json_out)
                ctx.panel.set_state("last_message", "Pegasus analysis complete.")
                ctx.ops.notify("Safety analysis complete.", variant="success")
            else:
                ctx.ops.notify(f"Unknown action: {action}", variant="warning")
        except Exception as exc:
            ctx.ops.notify(f"Action failed: {exc}", variant="warning")

    def render(self, ctx):
        dataset = ctx.dataset
        panel = types.Object()
        st = ctx.panel.state

        if not is_twelvelabs_configured(ctx):
            panel.view(
                "tl_fallback",
                types.Warning(
                    label="Twelve Labs not configured",
                    description=TWELVELABS_SETUP_MESSAGE,
                ),
            )
        else:
            panel.view(
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
            else "No index yet — run operator Twelve Labs index (create or link)."
        )
        panel.view(
            "idx_notice", types.Notice(label="Twelve Labs index", description=desc)
        )

        dd = types.Dropdown()
        dd.add_choice(
            "upload_selected",
            label="Upload all selected samples",
        )
        dd.add_choice(
            "upload_current",
            label="Upload current / first selected sample",
        )
        dd.add_choice(
            "view_embeddings",
            label="View embeddings (preview)",
        )
        dd.add_choice(
            "pegasus_safety",
            label="Pegasus safety analysis – current sample (JSON)",
        )
        dd.add_choice(
            "pegasus_safety_selected",
            label="Pegasus safety analysis – all selected samples (JSON)",
        )

        panel.enum(
            "panel_action",
            dd.values(),
            view=dd,
            label="Action",
            default=st.get("panel_action", "upload_selected"),
        )
        panel.str(
            "last_message",
            label="Status",
            allow_empty=True,
            view=types.ReadOnlyView(),
            default=st.get("last_message", ""),
        )
        panel.str(
            "safety_output",
            label="Pegasus output (JSON)",
            allow_empty=True,
            view=types.TextFieldView(read_only=True, multiline=True),
            default=st.get("safety_output", ""),
        )
        panel.btn(
            "run_btn",
            label="Run",
            variant="contained",
            on_click=self.method_to_uri("on_run_action"),
        )
        return types.Property(panel)
