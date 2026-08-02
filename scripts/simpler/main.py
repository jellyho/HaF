"""SimplerEnv (google_robot) closed-loop evaluation client for HaF policies.

Mirrors scripts/libero/main.py: this is a WebSocket CLIENT to a running `scripts/serve_policy.py` server.
It steps SimplerEnv google_robot tasks, sends observations to the policy server, and executes the returned
action chunk. Use it to evaluate the Exp-3 RT-1 models (exp3_rt1_{bc,langact,pred,mix,mix_ki}).

Pipeline:
  1) train:   python scripts/train.py exp3_rt1_mix --exp_name mix1        (PaliGemma init, RT-1 data)
  2) serve:   python scripts/serve_policy.py --policy.config=exp3_rt1_mix \
                     --policy.dir=checkpoints/exp3_rt1_mix/mix1/30000 --policy.type=flow
  3) eval:    python scripts/simpler/main.py --task-set visual_matching

Requires SimplerEnv installed: https://github.com/simpler-env/SimplerEnv  (pip install simpler-env, ManiSkill2).

NOTE (validate on first run): the RT-1 action convention (world_vector[3], rotation_delta[3], gripper[1]) is
what SimplerEnv google_robot consumes directly, and the proprio->state mapping mirrors rt1_dataset_transform
(eef pos + euler + gripper). Both are marked ASSUMPTION below — check against your training data transform.
"""
import collections
import dataclasses
import datetime
import enum
import json
import logging
import pathlib

import imageio
import numpy as np
from scipy.spatial.transform import Rotation as R
import tqdm
import tyro

# openpi_client is only needed for the live eval (websocket client + image tools); imported lazily inside
# eval_simpler so the module-level helpers (simpler_obs_to_state / rt1_action_to_simpler) can be reused by
# scripts/simpler/smoke_test.py in an env without openpi_client installed.

# SimplerEnv / ManiSkill2 imports are done lazily inside eval_simpler() so this file can be imported
# without the sim installed (e.g. for --help).

# google_robot task suites shipped with SimplerEnv.
GOOGLE_ROBOT_TASKS = {
    # "visual matching" variant-aggregation tasks (the standard SimplerEnv google_robot benchmark)
    "visual_matching": [
        "google_robot_pick_coke_can",
        "google_robot_move_near",
        "google_robot_open_drawer",
        "google_robot_close_drawer",
        "google_robot_place_apple_in_closed_top_drawer",
    ],
    # a quick smoke set
    "smoke": ["google_robot_pick_coke_can"],
}


class PolicyType(str, enum.Enum):
    LAP = "LAP"
    LAP_AR = "LAP_AR"


@dataclasses.dataclass
class Args:
    # --- model server ---
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    policy_type: PolicyType = PolicyType.LAP
    frame_description: str = "robot base frame"  # RT-1 was trained in the base frame

    # --- SimplerEnv ---
    task_set: str = "visual_matching"     # key into GOOGLE_ROBOT_TASKS
    num_trials_per_task: int = 25
    max_steps: int = 120                  # google_robot episodes are short-horizon
    seed: int = 0

    # --- utils ---
    video_out_path: str = "data/simpler/videos"
    results_out_path: str = "data/simpler/results"
    save_video: bool = True


def eval_simpler(args: Args) -> None:
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as _websocket_client_policy

    np.random.seed(args.seed)
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.results_out_path).mkdir(parents=True, exist_ok=True)

    tasks = GOOGLE_ROBOT_TASKS[args.task_set]
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logging.info(f"SimplerEnv task set '{args.task_set}': {tasks}")

    all_results = {
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "task_set": args.task_set, "policy_type": args.policy_type.value,
            "num_trials_per_task": args.num_trials_per_task, "replan_steps": args.replan_steps,
            "seed": args.seed,
        },
        "episodes": [], "per_task_results": [], "summary": {},
    }

    total_episodes, total_successes = 0, 0
    for task_name in tasks:
        env = simpler_env.make(task_name)
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=task_name):
            obs, reset_info = env.reset(seed=args.seed + episode_idx)
            instruction = env.get_language_instruction()
            action_plan = collections.deque()
            replay_images = []
            done, truncated, success = False, False, False
            t = 0
            while not (done or truncated) and t < args.max_steps:
                img_raw = get_image_from_maniskill2_obs_dict(env, obs)  # HxWx3 uint8
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img_raw, args.resize_size, args.resize_size))
                if not action_plan:
                    request = obs_to_request(obs, img, instruction, args.frame_description)
                    response = client.infer(request)
                    action_chunk = np.asarray(response["actions"], dtype=np.float32)
                    assert action_chunk.ndim == 2 and len(action_chunk) >= args.replan_steps, (
                        f"expected an action chunk of >= {args.replan_steps} steps, got {action_chunk.shape}")
                    action_plan.extend(action_chunk[: args.replan_steps])
                replay_images.append(img)
                action = action_plan.popleft()
                env_action = rt1_action_to_simpler(action)
                obs, reward, done, truncated, info = env.step(env_action)
                success = bool(info.get("success", done))
                if success:
                    done = True
                t += 1

            task_episodes += 1; total_episodes += 1
            if success:
                task_successes += 1; total_successes += 1
            all_results["episodes"].append({
                "task": task_name, "episode_id": episode_idx, "instruction": instruction,
                "success": success, "num_steps": t,
            })
            if args.save_video and replay_images:
                seg = task_name
                suffix = "success" if success else "failure"
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"{seg}_ep{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images], fps=10)
            logging.info(f"[{task_name}] ep {episode_idx}: success={success}  "
                         f"running {total_successes}/{total_episodes} ({100*total_successes/total_episodes:.1f}%)")

        rate = task_successes / task_episodes if task_episodes else 0.0
        all_results["per_task_results"].append(
            {"task": task_name, "num_episodes": task_episodes, "num_successes": task_successes, "success_rate": rate})
        logging.info(f"== {task_name}: {task_successes}/{task_episodes} = {rate:.3f} ==")

    overall = total_successes / total_episodes if total_episodes else 0.0
    all_results["summary"] = {"total_episodes": total_episodes, "total_successes": total_successes,
                              "overall_success_rate": overall, "num_tasks": len(tasks)}
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = pathlib.Path(args.results_out_path) / f"results_{args.task_set}_{args.policy_type.value}_{ts}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"OVERALL success rate: {overall:.3f}  ({total_successes}/{total_episodes})  -> {out}")


def obs_to_request(obs, img, instruction: str, frame_description: str):
    """Build the server request. Matches the LIBERO client's schema (single primary camera for RT-1)."""
    state = simpler_obs_to_state(obs)
    return {
        "observation": {
            "base_0_rgb": img,
            "state": state,
        },
        "prompt": str(instruction),
        "frame_description": frame_description,
    }


def simpler_obs_to_state(obs) -> np.ndarray:
    """ASSUMPTION: reproduce the RT-1 training state = eef_state[pos(3)+euler(3)] + gripper(1) (8-dim).

    rt1_dataset_transform builds eef_state from base_pose_tool_reached (position + euler-from-quaternion) and
    a gripper scalar. SimplerEnv google_robot exposes the tcp pose under obs["extra"]["tcp_pose"] as
    [x,y,z, qw,qx,qy,qz] (ManiSkill2 wxyz quaternion order) and gripper qpos under obs["agent"]["qpos"].
    VALIDATE the exact keys/quaternion order against your SimplerEnv version on first run.
    """
    # mini-VLA state = training cartt (base_pose_tool_reached, 7-d pos+quat) + gripper(1) = 8-d.
    # ASSUMPTION: SimplerEnv tcp_pose ([x,y,z, qw,qx,qy,qz], 7-d) matches cartt. VALIDATE on first rollout;
    # if success ~0, the mini-VLA is vision-dominant + z-scores state, so a state mismatch degrades gracefully.
    extra = obs.get("extra", {}) if isinstance(obs, dict) else {}
    tcp = np.asarray(extra.get("tcp_pose", np.zeros(7)), dtype=np.float32)
    if tcp.shape[0] < 7:
        tcp = np.concatenate([tcp, np.zeros(7 - tcp.shape[0], np.float32)])
    agent = obs.get("agent", {}) if isinstance(obs, dict) else {}
    qpos = np.asarray(agent.get("qpos", []), dtype=np.float32)
    gripper = np.array([np.clip(qpos[-1], 0.0, 1.0)] if qpos.size else [0.0], dtype=np.float32)
    return np.concatenate([tcp[:7], gripper]).astype(np.float32)   # 8-d


def rt1_action_to_simpler(action: np.ndarray) -> np.ndarray:
    """ASSUMPTION: the model outputs the RT-1 action [world_vector(3), rotation_delta(3), gripper(1)], which
    SimplerEnv google_robot consumes directly (delta xyz in meters, delta rotation, gripper action).
    Gripper: RT-1 uses closedness in [0,1]; SimplerEnv google_robot expects a gripper action where >0 tends to
    close. If success is near-zero on first run, flip the gripper sign / rescale here."""
    a = np.asarray(action, dtype=np.float32).copy()
    # pass-through arm delta; map gripper closedness[0,1] -> [-1,1] action (1=open, -1=close) is a common
    # convention; adjust after a smoke run.
    a[6] = 1.0 - 2.0 * np.clip(a[6], 0.0, 1.0)
    return a


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_simpler)
