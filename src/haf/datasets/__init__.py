"""RLDS datasets package.

This package provides dataset classes for loading and processing
robotics and VQA datasets in a standardized format.

Usage:
    from haf.datasets import get_dataset_class, register_dataset
    from haf.datasets import TrajectoryOutputBuilder, ObservationBuilder
    from haf.datasets import get_vqa_dataset_id, get_dataset_config
"""

# Base classes
from haf.datasets.base_dataset import BaseDataset
from haf.datasets.base_dataset import BaseRobotDataset
from haf.datasets.output_schema import ObservationBuilder
from haf.datasets.output_schema import TrajectoryOutputBuilder
from haf.datasets.registry import DATASET_REGISTRY  # Registry
from haf.datasets.registry import VQA_DATASET_NAMES
from haf.datasets.registry import WRIST_ROTATION_PATTERNS
from haf.datasets.registry import DatasetConfig  # Configuration (now in registry)
from haf.datasets.registry import DatasetMetadata
from haf.datasets.registry import get_action_bounds
from haf.datasets.registry import get_dataset_class
from haf.datasets.registry import get_dataset_class_with_fallback
from haf.datasets.registry import get_dataset_config
from haf.datasets.registry import get_dataset_metadata
from haf.datasets.registry import get_num_vqa_datasets
from haf.datasets.registry import get_tfds_name_with_version
from haf.datasets.registry import get_vqa_dataset_id
from haf.datasets.registry import get_vqa_dataset_name
from haf.datasets.registry import is_bimanual_dataset
from haf.datasets.registry import is_navigation_dataset
from haf.datasets.registry import is_vqa_dataset
from haf.datasets.registry import list_registered_datasets
from haf.datasets.registry import needs_wrist_rotation
from haf.datasets.registry import register_dataset
from haf.datasets.registry import register_dataset_config
from haf.datasets.registry import requires_hash_tables

__all__ = [
    # Registry
    "DATASET_REGISTRY",
    "VQA_DATASET_NAMES",
    "DatasetMetadata",
    "get_dataset_class",
    "get_dataset_class_with_fallback",
    "get_dataset_metadata",
    "get_num_vqa_datasets",
    "get_vqa_dataset_id",
    "get_vqa_dataset_name",
    "is_vqa_dataset",
    "list_registered_datasets",
    "register_dataset",
    "requires_hash_tables",
    # Configuration
    "DatasetConfig",
    "get_action_bounds",
    "get_dataset_config",
    "get_tfds_name_with_version",
    "is_bimanual_dataset",
    "is_navigation_dataset",
    "needs_wrist_rotation",
    "register_dataset_config",
    "WRIST_ROTATION_PATTERNS",
    # Output builders
    "ObservationBuilder",
    "TrajectoryOutputBuilder",
    # Base classes
    "BaseDataset",
    "BaseRobotDataset",
]
