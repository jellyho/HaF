"""Dataset-specific configuration system.

This module re-exports configuration utilities from the unified registry.
The actual configuration data and logic now lives in registry.py.

For new code, prefer importing directly from registry:
    from haf.datasets.registry import get_dataset_config, DatasetConfig
"""

# Re-export selected registry symbols.
from haf.datasets.registry import ALOHA_CONFIG
from haf.datasets.registry import BRIDGE_CONFIG
from haf.datasets.registry import DOBBE_CONFIG
from haf.datasets.registry import DROID_CONFIG  # Pre-defined configs
from haf.datasets.registry import FMB_CONFIG
from haf.datasets.registry import GNM_CONFIG
from haf.datasets.registry import LIBERO_CONFIG
from haf.datasets.registry import LVIS_CONFIG
from haf.datasets.registry import MOLMOACT_CONFIG
from haf.datasets.registry import WRIST_ROTATION_PATTERNS  # Constants
from haf.datasets.registry import DatasetConfig  # Configuration class
from haf.datasets.registry import get_action_bounds
from haf.datasets.registry import get_dataset_config  # Functions
from haf.datasets.registry import get_tfds_name_with_version
from haf.datasets.registry import is_bimanual_dataset
from haf.datasets.registry import is_navigation_dataset
from haf.datasets.registry import needs_wrist_rotation
from haf.datasets.registry import register_dataset_config

__all__ = [
    # Configuration class
    "DatasetConfig",
    # Pre-defined configs
    "DROID_CONFIG",
    "FMB_CONFIG",
    "DOBBE_CONFIG",
    "LVIS_CONFIG",
    "BRIDGE_CONFIG",
    "GNM_CONFIG",
    "LIBERO_CONFIG",
    "MOLMOACT_CONFIG",
    "ALOHA_CONFIG",
    # Constants
    "WRIST_ROTATION_PATTERNS",
    # Functions
    "get_dataset_config",
    "register_dataset_config",
    "get_tfds_name_with_version",
    "get_action_bounds",
    "needs_wrist_rotation",
    "is_navigation_dataset",
    "is_bimanual_dataset",
]
