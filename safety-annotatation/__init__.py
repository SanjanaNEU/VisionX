"""Safety annotation — FiftyOne plugin (Twelve Labs + EDA)."""

import os

from fiftyone.operators import Operator, OperatorConfig
import fiftyone.operators.types as types
from fiftyone.operators.panel import Panel, PanelConfig

from .twelve_labs_helpers import (
    analyze_safety,
    create_index,
    fetch_embedding_summary,
    get_client,
    upload_video_file,
)


class SummarizeSafetyLabels(Operator):
    """EDA: label counts for the current view."""

    @property
    def config(self):
        return OperatorConfig(
            name="summarize_safety_labels",
            label="Summarize safety labels (EDA)",
            description=(
                "Counts ground_truth.label on the current view and shows a notification."
            ),
        )

    def execute(self, ctx):
        view = ctx.view
        if view is None:
            ctx.ops.notify("No view in context.", variant="warning")
            return {}

        try:
            counts = view.count_values("ground_truth.label")
        except Exception as exc:
            ctx.ops.notify(
                f"Could not read ground_truth.label: {exc}",
                variant="warning",
            )
            return {}

        total = sum(counts.values())
        lines = [
            f"Samples in view: {len(view)}",
            f"Distinct labels: {len(counts)}",
            "",
        ]
        for label, c in sorted(counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * c / total if total else 0.0
            lines.append(f"  {label}: {c} ({pct:.1f}%)")

        ctx.ops.notify("\n".join(lines), variant="success")
        return {"counts": counts, "total": total}


class CreateTwelveLabsIndex(Operator):
    """Create a Twelve Labs index (Marengo + Pegasus) and store id on the dataset."""

    @property
    def config(self):
        return OperatorConfig(
            name="create_twelvelabs_index",
            label="Create Twelve Labs index",
            description=(
                "Creates a new index with marengo3.0 and pegasus1.2, saves tl_index_id to dataset.info."
            ),
            allow_delegated_execution=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str(
            "index_name",
            label="Index name",
            description="Name for the new Twelve Labs index.",
            required=True,
        )
        return types.Property(inputs)

    def execute(self, ctx):
        dataset = ctx.dataset
        if dataset is None:
            ctx.ops.notify("No dataset in context.", variant="warning")
            return {}

        name = (ctx.params.get("index_name") or "").strip()
        if not name:
            ctx.ops.notify("index_name is required.", variant="warning")
            return {}

        try:
            client = get_client(ctx)
            index_id = create_index(client, name)
        except Exception as exc:
            ctx.ops.notify(f"Create index failed: {exc}", variant="warning")
            return {}

        if dataset.info is None:
            dataset.info = {}
        dataset.info["tl_index_id"] = index_id
        dataset.save()
        ctx.ops.notify(f"Created index. id={index_id}", variant="success")
        return {"index_id": index_id}


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
    """Baseline panel: upload to TL index, embedding preview, Pegasus safety JSON."""

    @property
    def config(self):
        return PanelConfig(
            name="safety_annotation_panel",
            label="Safety + Twelve Labs",
            surfaces="grid",
        )

    def on_run_action(self, ctx):
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
                ctx.panel.set("last_message", msg)
                ctx.panel.set("embedding_preview", "")
                ctx.panel.set("safety_output", "")
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
                ctx.panel.set("embedding_preview", preview)
                ctx.panel.set("last_message", "Retrieved embedding preview.")
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
                ctx.panel.set("safety_output", text_out)
                ctx.panel.set("last_message", "Pegasus analysis complete.")
                ctx.ops.notify("Safety analysis complete.", variant="success")
            else:
                ctx.ops.notify(f"Unknown action: {action}", variant="warning")
        except Exception as exc:
            ctx.ops.notify(f"Action failed: {exc}", variant="warning")

    def render(self, ctx):
        dataset = ctx.dataset
        idx = (dataset.info or {}).get("tl_index_id") if dataset else None
        desc = (
            f"tl_index_id: {idx}"
            if idx
            else "No index yet — run operator Create Twelve Labs index."
        )
        notice = types.Notice(label="Twelve Labs index", description=desc)

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

        panel = types.Object()
        panel.view("idx_notice", notice)
        panel.enum(
            "panel_action",
            dd.values(),
            view=dd,
            label="Action",
            default=ctx.panel.state.get("panel_action", "upload_current"),
        )
        panel.str(
            "last_message",
            label="Status",
            view=types.ReadOnlyView(),
            default=ctx.panel.state.get("last_message", ""),
        )
        panel.str(
            "embedding_preview",
            label="Embedding preview",
            view=types.ReadOnlyView(),
            default=ctx.panel.state.get("embedding_preview", ""),
        )
        panel.str(
            "safety_output",
            label="Pegasus output",
            view=types.ReadOnlyView(),
            default=ctx.panel.state.get("safety_output", ""),
        )
        panel.btn(
            "run_btn",
            label="Run",
            variant="contained",
            on_click=self.method_to_uri("on_run_action"),
        )
        return types.Property(panel)


def register(pctx):
    pctx.register(SummarizeSafetyLabels)
    pctx.register(CreateTwelveLabsIndex)
    pctx.register(SafetyAnnotationPanel)
