"""Create a Twelve Labs index (Marengo + Pegasus)."""

from fiftyone.operators import Operator, OperatorConfig
import fiftyone.operators.types as types

from ..twelve_labs_helpers import (
    TWELVELABS_SETUP_MESSAGE,
    create_index,
    get_client,
    is_twelvelabs_configured,
)


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
        inputs.str(
            "index_name",
            label="Index name",
            description="Name for the new Twelve Labs index (ignored until API key is set).",
            required=True,
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
