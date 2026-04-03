"""Twelve Labs API helpers for the safety-annotation plugin."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from twelvelabs import TwelveLabs, IndexesCreateRequestModelsItem, ResponseFormat

TWELVELABS_SETUP_MESSAGE = (
    "Twelve Labs API key not found. Add TWELVELABS_API_KEY using one of:\n"
    "• FiftyOne App: Settings → Plugin secrets (key name: TWELVELABS_API_KEY)\n"
    "• Environment: export TWELVELABS_API_KEY=... (or TWELVE_LABS_API_KEY)\n"
    "• .env loaded before starting the App / notebook\n"
    "This plugin declares the secret in fiftyone.yml."
)


def is_twelvelabs_configured(ctx) -> bool:
    """True if a Twelve Labs API key is available (secrets or env)."""
    return bool(get_api_key(ctx))


def twelvelabs_config_error_message(ctx) -> Optional[str]:
    """None if configured; otherwise a user-facing explanation."""
    if is_twelvelabs_configured(ctx):
        return None
    return TWELVELABS_SETUP_MESSAGE


def get_api_key(ctx) -> Optional[str]:
    """Resolve API key from FiftyOne plugin secrets or environment."""
    key = None
    if ctx is not None:
        try:
            key = ctx.secret("TWELVELABS_API_KEY")
        except Exception:
            pass
    if not key:
        key = os.getenv("TWELVELABS_API_KEY") or os.getenv("TWELVE_LABS_API_KEY")
    return key


def get_client(ctx):
    """Build a TwelveLabs client; raises if no API key."""
    api_key = get_api_key(ctx)
    if not api_key:
        raise ValueError(
            "Missing TWELVELABS_API_KEY (set in App secrets / environment)."
        )
    return TwelveLabs(api_key=api_key)


def create_index(client: TwelveLabs, index_name: str) -> str:
    """Create an index with Marengo 3.0 + Pegasus 1.2; return index id."""
    resp = client.indexes.create(
        index_name=index_name,
        models=[
            IndexesCreateRequestModelsItem(
                model_name="marengo3.0",
                model_options=["visual", "audio"],
            ),
            IndexesCreateRequestModelsItem(
                model_name="pegasus1.2",
                model_options=["visual", "audio"],
            ),
        ],
    )
    index_id = getattr(resp, "id", None) or getattr(resp, "_id", None)
    if not index_id:
        raise RuntimeError("Create index response missing id.")
    return index_id


def verify_index_exists(client: TwelveLabs, index_id: str) -> str:
    """
    Confirm the index id is reachable with the current API key; return normalized id.
    Raises if the index is missing or the request fails.
    """
    s = (index_id or "").strip()
    if not s:
        raise ValueError("Index id is empty.")
    client.indexes.retrieve(s)
    return s


def upload_video_file(client: TwelveLabs, index_id: str, filepath: str) -> str:
    """Upload a local file to the index, wait until ready; return Twelve Labs video_id."""
    with open(filepath, "rb") as f:
        task = client.tasks.create(index_id=index_id, video_file=f)
    task_id = getattr(task, "id", None) or getattr(task, "_id", None)
    if not task_id:
        raise RuntimeError("Task create response missing id.")

    completed = client.tasks.wait_for_done(task_id=task_id, sleep_interval=5.0)
    status = getattr(completed, "status", None)
    if status == "failed":
        raise RuntimeError("Video indexing task failed.")
    video_id = getattr(completed, "video_id", None)
    if not video_id:
        raise RuntimeError("Indexed task missing video_id.")
    return video_id


def fetch_embedding_summary(client: TwelveLabs, index_id: str, video_id: str) -> str:
    """Return a short Marengo embedding preview (not full vectors)."""
    r = client.indexes.videos.retrieve(
        index_id,
        video_id,
        embedding_option=["visual"],
    )
    emb = r.embedding
    if not emb or not emb.video_embedding or not emb.video_embedding.segments:
        return "No embedding segments returned (is the video indexed with Marengo?)."
    segments = emb.video_embedding.segments
    seg = segments[0]
    floats = getattr(seg, "float_", None) or getattr(seg, "float", None) or []
    dim = len(floats)
    head = floats[:8]
    return f"segments={len(segments)}, dim={dim}, first_8={head!r}"


def analyze_safety(
    client: TwelveLabs,
    video_id: str,
    ground_truth_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Pegasus: structured JSON with reasoning and danger_score 1–10.
    Falls back to raw text if JSON parsing fails.
    """
    label_hint = (
        f"The dataset label for this clip is: {ground_truth_label!r}. "
        if ground_truth_label
        else ""
    )
    prompt = (
        label_hint
        + "Watch the video and explain in detail whether the behavior shown is unsafe "
        "for an industrial / workplace safety context. Reference timing when possible "
        "(e.g. early vs late in the clip). "
        "Respond ONLY as JSON matching the schema: reasoning (string), danger_score (integer 1-10)."
    )
    schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "danger_score": {"type": "integer"},
        },
        "required": ["reasoning", "danger_score"],
    }
    resp = client.analyze(
        video_id=video_id,
        prompt=prompt,
        response_format=ResponseFormat(type="json_schema", json_schema=schema),
        temperature=0.2,
        max_tokens=2000,
    )
    raw = resp.data
    if raw is None:
        return {"error": "empty_response", "raw": None}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw, "parse_error": True}
    return {"raw": str(raw)}
