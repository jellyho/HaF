"""Backward-compatible exports for OpenPI-style policy transforms.

The canonical transform implementations live in `lap.policies.transforms`.
This module remains as a stable import path for existing code.
"""

from haf.policies.transforms.input_transforms import CoTInputs
from haf.policies.transforms.output_transforms import CoTOutputs

__all__ = ["CoTInputs", "CoTOutputs"]
