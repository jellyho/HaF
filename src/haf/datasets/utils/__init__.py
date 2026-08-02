"""Dataset utilities for RLDS datasets.

This package provides utilities for working with robot learning datasets
including transforms, configurations, image processing, and rotation utilities.
"""

# Re-export commonly used items for convenience
# Dataset configs and mixtures
from haf.datasets.utils.configs import OXE_DATASET_CONFIGS
from haf.datasets.utils.configs import OXE_DATASET_METADATA

# Constants
from haf.datasets.utils.constants import DATASETS_REQUIRING_WRIST_ROTATION
from haf.datasets.utils.constants import DEFAULT_IMAGE_RESOLUTION
from haf.datasets.utils.constants import EPSILON
from haf.datasets.utils.constants import FALLBACK_INSTRUCTIONS
from haf.datasets.utils.constants import GRIPPER_BINARIZE_THRESHOLD
from haf.datasets.utils.constants import GRIPPER_OPEN_THRESHOLD
from haf.datasets.utils.dataset_discovery import ensure_datasets_registered

# Encoding types
from haf.datasets.utils.helpers import ActionEncoding
from haf.datasets.utils.helpers import NormalizationType
from haf.datasets.utils.helpers import StateEncoding
from haf.datasets.utils.helpers import state_encoding_to_type
from haf.datasets.utils.image_utils import make_decode_images_fn
from haf.datasets.utils.image_utils import tf_maybe_rotate_180

# Image utilities
from haf.datasets.utils.image_utils import tf_rotate_180
from haf.datasets.utils.mixtures import OXE_NAMED_MIXTURES
from haf.datasets.utils.normalization_and_config import allocate_threads
from haf.datasets.utils.normalization_and_config import load_dataset_kwargs

# Data utilities
from haf.datasets.utils.normalization_and_config import normalize_action_and_proprio
from haf.datasets.utils.normalization_and_config import pprint_data_mixture
from haf.datasets.utils.rotation_utils import apply_coordinate_transform
from haf.datasets.utils.rotation_utils import axis_angle_to_euler
from haf.datasets.utils.rotation_utils import axis_angle_to_r6
from haf.datasets.utils.rotation_utils import coordinate_transform_bcz
from haf.datasets.utils.rotation_utils import coordinate_transform_dobbe
from haf.datasets.utils.rotation_utils import coordinate_transform_jaco
from haf.datasets.utils.rotation_utils import euler_diff
from haf.datasets.utils.rotation_utils import euler_to_quaternion
from haf.datasets.utils.rotation_utils import euler_to_r6

# Rotation utilities
from haf.datasets.utils.rotation_utils import euler_to_rotation_matrix
from haf.datasets.utils.rotation_utils import matrix_to_xyzrpy
from haf.datasets.utils.rotation_utils import quaternion_to_euler
from haf.datasets.utils.rotation_utils import quaternion_to_rotation_matrix
from haf.datasets.utils.rotation_utils import r6_to_euler
from haf.datasets.utils.rotation_utils import r6_to_rotation_matrix
from haf.datasets.utils.rotation_utils import rotation_matrix_to_euler
from haf.datasets.utils.rotation_utils import rotation_matrix_to_quaternion
from haf.datasets.utils.rotation_utils import rotation_matrix_to_r6
from haf.datasets.utils.rotation_utils import wxyz_to_r6
from haf.datasets.utils.rotation_utils import zxy_to_xyz

# Statistics and discovery utilities
from haf.datasets.utils.statistics import GlobalStatisticsBuilder
from haf.datasets.utils.tfdata_pipeline import dataset_size
from haf.datasets.utils.tfdata_pipeline import gather_with_last_value_padding

# Dataset utilities
from haf.datasets.utils.tfdata_pipeline import gather_with_padding
from haf.datasets.utils.tfdata_pipeline import prepare_batched_dataset
from haf.datasets.utils.transform_helpers import binarize_gripper_actions
from haf.datasets.utils.transform_helpers import build_matrix_state_transform
from haf.datasets.utils.transform_helpers import build_standard_eef_transform

# Transform helpers
from haf.datasets.utils.transform_helpers import compute_padded_movement_actions
from haf.datasets.utils.transform_helpers import extract_state_from_matrix
from haf.datasets.utils.transform_helpers import fill_empty_language_instruction
from haf.datasets.utils.transform_helpers import invert_gripper_actions
from haf.datasets.utils.transform_helpers import rel2abs_gripper_actions
from haf.datasets.utils.transform_helpers import rescale_action_with_bound

# Transform registry
from haf.datasets.utils.transforms import OXE_STANDARDIZATION_TRANSFORMS

__all__ = [
    # Mixtures
    "OXE_NAMED_MIXTURES",
    # Constants
    "DATASETS_REQUIRING_WRIST_ROTATION",
    "FALLBACK_INSTRUCTIONS",
    "DEFAULT_IMAGE_RESOLUTION",
    "GRIPPER_OPEN_THRESHOLD",
    "GRIPPER_BINARIZE_THRESHOLD",
    "EPSILON",
    # Encoding types
    "ActionEncoding",
    "StateEncoding",
    "NormalizationType",
    "state_encoding_to_type",
    # Configs
    "OXE_DATASET_CONFIGS",
    "OXE_DATASET_METADATA",
    # Rotation utilities
    "euler_to_rotation_matrix",
    "rotation_matrix_to_euler",
    "euler_to_quaternion",
    "quaternion_to_euler",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "rotation_matrix_to_r6",
    "r6_to_rotation_matrix",
    "euler_to_r6",
    "r6_to_euler",
    "apply_coordinate_transform",
    "coordinate_transform_bcz",
    "coordinate_transform_dobbe",
    "coordinate_transform_jaco",
    "euler_diff",
    "zxy_to_xyz",
    "matrix_to_xyzrpy",
    "axis_angle_to_r6",
    "axis_angle_to_euler",
    "wxyz_to_r6",
    # Transform helpers
    "compute_padded_movement_actions",
    "extract_state_from_matrix",
    "fill_empty_language_instruction",
    "binarize_gripper_actions",
    "invert_gripper_actions",
    "rel2abs_gripper_actions",
    "build_standard_eef_transform",
    "build_matrix_state_transform",
    "rescale_action_with_bound",
    # Image utilities
    "tf_rotate_180",
    "tf_maybe_rotate_180",
    "make_decode_images_fn",
    # Data utilities
    "normalize_action_and_proprio",
    "load_dataset_kwargs",
    "pprint_data_mixture",
    "allocate_threads",
    # Dataset utilities
    "gather_with_padding",
    "gather_with_last_value_padding",
    "dataset_size",
    "prepare_batched_dataset",
    # Transforms
    "OXE_STANDARDIZATION_TRANSFORMS",
    # Statistics
    "GlobalStatisticsBuilder",
    "ensure_datasets_registered",
]
