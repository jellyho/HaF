"""Standardized output schema builders for datasets.

This module provides factory functions to construct consistent output
dictionaries for robot and VQA datasets, reducing code duplication
and ensuring all required fields are present.
"""

from __future__ import annotations

import tensorflow as tf


class TrajectoryOutputBuilder:
    """Builder for consistent trajectory-level output dictionaries.

    Use these methods in apply_restructure() to create standardized
    output dictionaries with all required fields.
    """

    @staticmethod
    def retro_fields(trajectory_id, state, traj_len, control_frequency, seed):
        """Per-frame RETROSPECTIVE / introspective fields for the recoverability-lowering aux questions
        (PROGRESS_ESTIMATION, DISPLACEMENT_FROM_START, PAST_GRIPPER_RECALL). All shape [traj_len].

        The START ANCHOR is sampled once per trajectory from the first ~2 s window (per user spec), seeded by
        the trajectory id for determinism. Assumes an eef-pose state layout: pos = state[:, :3],
        gripper = state[:, -1]. For non-eef state types displacement/gripper are approximate but harmless
        (these fields are only read when the retro question types are enabled).
        """
        tl = tf.cast(traj_len, tf.float32)
        progress = tf.range(traj_len, dtype=tf.float32) / tf.maximum(tl - 1.0, 1.0)

        state = tf.cast(state, tf.float32)
        pos = state[:, :3]   # (T, ≤3) — TF clips if state_dim < 3
        grip = state[:, -1]  # (T,)

        win = tf.maximum(tf.minimum(tf.cast(2.0 * float(control_frequency), tf.int32), traj_len), 1)
        tid_hash = tf.strings.to_hash_bucket_fast(tf.reshape(trajectory_id, []), 2147483647)
        start_idx = tf.random.stateless_uniform([], seed=[int(seed) + tid_hash, 991], minval=0, maxval=win, dtype=tf.int32)

        start_pos = pos[start_idx]                                   # (≤3,)
        displacement_cm = 100.0 * tf.norm(pos - start_pos[None, :], axis=1)  # (T,)
        start_gripper = tf.fill([traj_len], grip[start_idx])         # (T,)
        return {"progress": progress, "displacement_cm": displacement_cm, "start_gripper": start_gripper}

    @staticmethod
    def build_robot_trajectory(
        observation: dict,
        actions: tf.Tensor,
        language_action: tf.Tensor,
        prompt: tf.Tensor,
        trajectory_id: tf.Tensor,
        dataset_name: str,
        traj_len: tf.Tensor | int,
        *,
        is_bimanual: bool = False,
        state_type: str = "eef_pose",
        has_wrist_image: bool = True,
        needs_wrist_rotation: bool = False,
        is_navigation: bool = False,
        extra_fields: dict | None = None,
    ) -> dict:
        """Build standardized robot trajectory output.

        Args:
            observation: Dict containing image and state observations.
            actions: Action tensor of shape [T, action_dim].
            language_action: Language action tensor of shape [T, lang_action_dim].
            prompt: Language instruction tensor (scalar or [T]).
            trajectory_id: Trajectory identifier tensor.
            dataset_name: Name of the dataset (string).
            traj_len: Length of the trajectory (int or scalar tensor).
            is_bimanual: Whether this is a bimanual robot.
            state_type: Type of state encoding ("none", "joint_pos", "eef_pose").
            has_wrist_image: Whether wrist image is available.
            needs_wrist_rotation: Whether wrist image needs 180° rotation.
            is_navigation: Whether this is a navigation dataset.
            extra_fields: Additional fields to include in output.

        Returns:
            Standardized trajectory dictionary.
        """
        output = {
            "observation": observation,
            "actions": tf.cast(actions, tf.float32),
            "language_action": tf.cast(language_action, tf.float32),
            "language_instruction": prompt,  # Will be renamed to "prompt" in repack
            "trajectory_id": trajectory_id,
            "dataset_name": tf.fill([traj_len], tf.constant(dataset_name)),
            "is_bimanual": tf.fill([traj_len], tf.constant(is_bimanual)),
            "state_type": tf.fill([traj_len], tf.constant(state_type)),
            "raw_state": tf.identity(observation["state"]),
            "is_navigation": tf.fill([traj_len], tf.constant(is_navigation)),
            "has_wrist_image": tf.fill([traj_len], tf.constant(has_wrist_image)),
            "needs_wrist_rotation": tf.fill([traj_len], tf.constant(needs_wrist_rotation)),
        }

        if extra_fields:
            output.update(extra_fields)

        return output

    @staticmethod
    def build_vqa_frame(
        observation: dict,
        prompt: tf.Tensor,
        caption: tf.Tensor,
        dataset_name: str,
        vqa_dataset_id: int,
        action_horizon: int,
        action_dim: int,
        state_dim: int,
        *,
        has_wrist_image: bool = False,
        needs_wrist_rotation: bool = False,
        extra_fields: dict | None = None,
    ) -> dict:
        """Build standardized VQA frame output.

        Args:
            observation: Dict containing image and state observations.
            prompt: The question/prompt tensor.
            caption: The answer/caption tensor.
            dataset_name: Name of the dataset (string).
            vqa_dataset_id: Integer ID for VQA dataset metrics.
            action_horizon: Action horizon for padding.
            action_dim: Action dimension for padding.
            state_dim: State dimension for padding.
            has_wrist_image: Whether wrist image is available.
            needs_wrist_rotation: Whether wrist image needs 180° rotation.
            extra_fields: Additional fields to include in output.

        Returns:
            Standardized VQA frame dictionary.
        """
        output = {
            "observation": observation,
            "prompt": prompt,
            "caption": caption,
            "dataset_name": tf.constant(dataset_name, dtype=tf.string),
            "time_horizon_seconds": tf.constant(1.0, dtype=tf.float32),
            "is_bimanual": tf.constant(False, dtype=tf.bool),
            "state_type": tf.constant("none", dtype=tf.string),
            "is_vqa_sample": tf.constant(True, dtype=tf.bool),
            "is_prediction_sample": tf.constant(False, dtype=tf.bool),
            "pred_use_primary": tf.constant(False, dtype=tf.bool),
            "raw_state": tf.zeros([state_dim], dtype=tf.float32),
            "is_navigation": tf.constant(False, dtype=tf.bool),
            "has_wrist_image": tf.constant(has_wrist_image, dtype=tf.bool),
            "needs_wrist_rotation": tf.constant(needs_wrist_rotation, dtype=tf.bool),
            "vqa_dataset_id": tf.constant(vqa_dataset_id, dtype=tf.int32),
            "actions": tf.zeros([action_horizon, action_dim], dtype=tf.float32),
            "language_actions": tf.zeros([7], dtype=tf.float32),
            # retrospective-aux fields (kept for batch-key consistency with robot samples)
            "progress": tf.constant(0.0, dtype=tf.float32),
            "displacement_cm": tf.constant(0.0, dtype=tf.float32),
            "start_gripper": tf.constant(0.0, dtype=tf.float32),
        }

        if extra_fields:
            output.update(extra_fields)

        return output


class ObservationBuilder:
    """Builder for observation dictionaries."""

    @staticmethod
    def build_from_image_keys(
        old_obs: dict,
        image_obs_keys: dict[str, str | None],
        state_obs_keys: list[str | None] | None,
        traj_len: tf.Tensor | int,
        primary_image_key: str,
        wrist_image_key: str,
        wrist_image_right_key: str | None = None,
    ) -> dict:
        """Build observation dict from raw observation and key mappings.

        Args:
            old_obs: Original observation dictionary from RLDS.
            image_obs_keys: Mapping from standard keys (primary, wrist, wrist_right)
                           to dataset-specific keys.
            state_obs_keys: List of state observation keys to concatenate.
            traj_len: Length of the trajectory.
            primary_image_key: Target key for primary image.
            wrist_image_key: Target key for wrist image.
            wrist_image_right_key: Target key for right wrist image (optional).

        Returns:
            Standardized observation dictionary.
        """
        new_obs = {}

        # Map image keys
        key_mapping = {
            "primary": primary_image_key,
            "wrist": wrist_image_key,
        }
        if wrist_image_right_key:
            key_mapping["wrist_right"] = wrist_image_right_key

        for std_key, target_key in key_mapping.items():
            source_key = image_obs_keys.get(std_key)
            if source_key is None or source_key not in old_obs:
                new_obs[target_key] = tf.repeat("", traj_len)
            else:
                new_obs[target_key] = old_obs[source_key]

        # Build state
        if state_obs_keys:
            state_parts = [tf.cast(old_obs[k], tf.float32) for k in state_obs_keys if k is not None and k in old_obs]
            if state_parts:
                new_obs["state"] = tf.concat(state_parts, axis=1)
            else:
                new_obs["state"] = tf.zeros((traj_len, 0), dtype=tf.float32)
        else:
            new_obs["state"] = tf.zeros((traj_len, 0), dtype=tf.float32)

        return new_obs

    @staticmethod
    def build_vqa_observation(
        image_encoded: tf.Tensor,
        primary_image_key: str,
        wrist_image_key: str,
        state_dim: int,
        wrist_image_right_key: str | None = None,
    ) -> dict:
        """Build observation dict for VQA datasets.

        Args:
            image_encoded: Encoded image bytes.
            primary_image_key: Target key for primary image.
            wrist_image_key: Target key for wrist image.
            state_dim: Dimension of dummy state tensor.
            wrist_image_right_key: Target key for right wrist image (for consistency with robot datasets).

        Returns:
            Standardized VQA observation dictionary.
        """
        obs = {
            primary_image_key: image_encoded,
            wrist_image_key: tf.constant("", dtype=tf.string),
            "state": tf.zeros([state_dim], dtype=tf.float32),
        }
        # Include right wrist image key for consistency with robot datasets when mixing
        if wrist_image_right_key:
            obs[wrist_image_right_key] = tf.constant("", dtype=tf.string)
        return obs
