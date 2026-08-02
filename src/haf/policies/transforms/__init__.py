"""Transform utilities for CoT policy inputs and outputs.

This package intentionally avoids eager imports to prevent circular import
chains when individual transform submodules are imported.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ActionProcessor",
    "CoTInputs",
    "CoTOutputs",
    "ImageHandler",
    "PredictionSampleHandler",
    "RobotSampleHandler",
    "TextParser",
    "VQASampleHandler",
]

_MODULE_BY_EXPORT = {
    "ActionProcessor": "haf.policies.transforms.action_processor",
    "CoTInputs": "haf.policies.transforms.input_transforms",
    "CoTOutputs": "haf.policies.transforms.output_transforms",
    "ImageHandler": "haf.policies.transforms.image_handler",
    "PredictionSampleHandler": "haf.policies.transforms.sample_handlers",
    "RobotSampleHandler": "haf.policies.transforms.sample_handlers",
    "TextParser": "haf.policies.transforms.text_utils",
    "VQASampleHandler": "haf.policies.transforms.sample_handlers",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve package-level exports."""
    if name not in _MODULE_BY_EXPORT:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_MODULE_BY_EXPORT[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
