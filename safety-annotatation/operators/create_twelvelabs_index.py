"""Create a Twelve Labs index (Marengo + Pegasus)."""

from fiftyone.operators import Operator, OperatorConfig
import fiftyone.operators.types as types

from ..twelve_labs_helpers import (
    TWELVELABS_SETUP_MESSAGE,
    create_index,
    get_client,
    is_twelvelabs_configured,
    verify_index_exists,
)


class CreateTwelveLabsIndex(Operator):
    """Create or link a Twelve Labs index; store ``tl_index_id`` on the dataset."""

    @property
    def config(self):
        return OperatorConfig(
            name="create_twelvelabs_index",
            label="Twelve Labs index",
            description=(
                "Create a new Marengo + Pegasus index, or link an existing index id; "
                "saves tl_index_id to dataset.info."
            ),
            allow_delegated_execution=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        if is_twelvelabs_configured(ctx):
            inputs.view(
                "tl_cfg_ok",
                types.Notice(
                    label="Twelve Labs",
                    description="API key found (secret or environment).",
                ),
            )
        else:
            inputs.view(
                "tl_cfg_missing",
                types.Warning(
                    label="Twelve Labs API key missing",
                    description=TWELVELABS_SETUP_MESSAGE,
                ),
            )
        mode = types.RadioGroup()
        mode.add_choice(
            "create_new",
            label="Create new index",
            description="Marengo 3.0 + Pegasus 1.2 (visual + audio).",
        )
        mode.add_choice(
            "use_existing",
            label="Use existing index ID",
            description="Paste an id from the Twelve Labs dashboard (same API key).",
        )
        inputs.enum(
            "index_mode",
            mode.values(),
            view=mode,
            label="Index setup",
            required=True,
            default="create_new",
        )
        inputs.str(
            "index_name",
            label="New index name",
            description="Required when creating a new index.",
            allow_empty=True,
        )
        inputs.str(
            "existing_index_id",
            label="Existing index ID",
            description="Required when using an existing index (e.g. 65a1b2c3d4e5f6789012345).",
            allow_empty=True,
        )
        return types.Property(inputs)

    def execute(self, ctx):
        if not is_twelvelabs_configured(ctx):
            ctx.ops.notify(TWELVELABS_SETUP_MESSAGE, variant="warning")
            return {"error": "twelvelabs_not_configured"}

        dataset = ctx.dataset
        if dataset is None:
            ctx.ops.notify("No dataset in context.", variant="warning")
            return {}

        mode = (ctx.params.get("index_mode") or "create_new").strip()
        try:
            client = get_client(ctx)
            if mode == "use_existing":
                raw_id = (ctx.params.get("existing_index_id") or "").strip()
                if not raw_id:
                    ctx.ops.notify(
                        "Existing index ID is required for this option.",
                        variant="warning",
                    )
                    return {}
                index_id = verify_index_exists(client, raw_id)
                msg = f"Linked index. id={index_id}"
            else:
                name = (ctx.params.get("index_name") or "").strip()
                if not name:
                    ctx.ops.notify(
                        "Index name is required when creating a new index.",
                        variant="warning",
                    )
                    return {}
                index_id = create_index(client, name)
                msg = f"Created index. id={index_id}"
        except Exception as exc:
            ctx.ops.notify(f"Index setup failed: {exc}", variant="warning")
            return {}

        if dataset.info is None:
            dataset.info = {}
        dataset.info["tl_index_id"] = index_id
        dataset.save()
        ctx.ops.notify(msg, variant="success")
        return {"index_id": index_id}
