"""Safety annotation — FiftyOne plugin (Twelve Labs)."""

from .operators import CreateTwelveLabsIndex
from .panels import SafetyAnnotationPanel


def register(pctx):
    pctx.register(CreateTwelveLabsIndex)
    pctx.register(SafetyAnnotationPanel)
