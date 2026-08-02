"""Smoke test for the retrospective / task-inference question wiring (unit level, no RLDS needed).

Validates the LOGIC of the new pieces before any training run:
  1. TrajectoryOutputBuilder.retro_fields — shapes + value ranges on a synthetic trajectory.
  2. The retrospective answer generators (progress phase / displacement bucket / past gripper).
  3. PredictionSampleHandler question generation for the new types (prompt, answer) via synthetic `data`.

NOTE: this does NOT test the full tf.data plumbing (does `progress`/`displacement_cm`/`start_gripper` survive
the restructure→prediction→repack→batch path to the handler). For that, load a real batch with
`question_type_weights` set to the retro preset and `enable_prediction_training=True, pred_prob>0`, then print a
few (prompt, answer) pairs. RETRO_MOTION is NOT active (needs the retrospective image pair — see RECOVERABILITY.md).

Run:  .venv/bin/python scripts/test_retro_wiring.py
"""
import numpy as np
import tensorflow as tf

from haf.datasets.output_schema import TrajectoryOutputBuilder
from haf.policies import question_types as q


def test_retro_fields():
    T, D = 40, 7
    # synthetic eef-pose state: pos ramps, gripper flips halfway
    pos = np.cumsum(np.full((T, 3), 0.01, np.float32), axis=0)
    rest = np.zeros((T, 3), np.float32)
    grip = (np.arange(T) > 20).astype(np.float32)[:, None]
    state = tf.constant(np.concatenate([pos, rest, grip], axis=1))
    out = TrajectoryOutputBuilder.retro_fields(
        trajectory_id=tf.constant("traj_007"), state=state, traj_len=T, control_frequency=3, seed=0)
    prog, disp, sg = out["progress"].numpy(), out["displacement_cm"].numpy(), out["start_gripper"].numpy()
    assert prog.shape == (T,) and abs(prog[0]) < 1e-6 and abs(prog[-1] - 1.0) < 1e-6, prog[[0, -1]]
    assert disp.shape == (T,) and (disp >= -1e-4).all() and disp[-1] > disp[0], (disp[0], disp[-1])
    assert sg.shape == (T,) and set(np.unique(sg)).issubset({0.0, 1.0})
    print(f"[1] retro_fields OK  progress[0,-1]={prog[0]:.2f},{prog[-1]:.2f}  disp[-1]={disp[-1]:.1f}cm  start_gripper={sg[0]:.0f}")


def test_generators():
    assert q.compute_progress_phase(0.05) == "just started"
    assert q.compute_progress_phase(0.6) == "about halfway"
    assert q.compute_progress_phase(0.97) == "almost done"
    assert q.compute_displacement_magnitude(1.0) == "barely moved"
    assert "large" in q.compute_displacement_magnitude(40.0)
    assert q.compute_past_gripper(0.0, 1.0) == "the gripper opened earlier"
    print("[2] retro answer generators OK")


def test_handler():
    from haf.policies.transforms.sample_handlers import PredictionSampleHandler
    cfg = q.QuestionConfig(type_weights={t: 1.0 for t in [
        q.QuestionType.PROGRESS_ESTIMATION.value, q.QuestionType.DISPLACEMENT_FROM_START.value,
        q.QuestionType.PAST_GRIPPER_RECALL.value, q.QuestionType.TASK_INFERENCE.value]})
    # the retrospective branches do not use action_processor (only delta/task_prediction do) -> None is fine
    h = PredictionSampleHandler(question_config=cfg, action_processor=None)
    data = {"prompt": "pick up the coke can", "progress": 0.6, "displacement_cm": 12.0, "start_gripper": 0.0}
    mc = dict(dx_cm=2.0, dy_cm=-1.0, dz_cm=0.0, droll_deg=0, dpitch_deg=0, dyaw_deg=0, gripper=1.0)
    rng = np.random.default_rng(0)
    for qt in [q.QuestionType.PROGRESS_ESTIMATION, q.QuestionType.DISPLACEMENT_FROM_START,
               q.QuestionType.PAST_GRIPPER_RECALL, q.QuestionType.TASK_INFERENCE]:
        prompt, ans = h._format_question_answer(
            data=data, inputs={}, question_type=qt, motion_components=mc,
            dataset_name="fractal", initial_state=np.zeros(7), frame_description="robot base frame", rng=rng)
        print(f"[3] {qt.value:22s} Q={prompt[:42]!r:44s} A={ans!r}")
    print("[3] handler question generation OK")


if __name__ == "__main__":
    test_retro_fields()
    test_generators()
    test_handler()
    print("\nALL SMOKE CHECKS PASSED (unit level). Validate full tf.data plumbing with a real batch before training.")
