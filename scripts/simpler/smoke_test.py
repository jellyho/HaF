"""Standalone smoke test for the SimplerEnv eval harness — NO policy server needed.

Runs a google_robot task with RANDOM actions to verify, before we have a trained model, that:
  * simpler_env.make / reset / step / rendering work in this environment,
  * get_image_from_maniskill2_obs_dict returns a sane HxWx3 image,
  * our harness helpers simpler_obs_to_state() and rt1_action_to_simpler() run and produce right-shaped output,
  * we can read the true obs structure to VALIDATE the 3 ASSUMPTION spots in main.py
    (tcp_pose key + quaternion order, gripper qpos, image key).

Run (needs a GPU for SAPIEN rendering — launch via srun on an L40S node, do not grab a GPU directly):
    srun --partition=debug --gres=gpu:L40S:1 --time=00:15:00 \
        /data5/jellyho/Hindsight/simpler_venv/bin/python scripts/simpler/smoke_test.py --task google_robot_pick_coke_can
"""
import argparse
import numpy as np


def pretty(obs, prefix=""):
    if isinstance(obs, dict):
        for k, v in obs.items():
            if isinstance(v, dict):
                print(f"{prefix}{k}/")
                pretty(v, prefix + "  ")
            else:
                arr = np.asarray(v)
                print(f"{prefix}{k}: shape={arr.shape} dtype={arr.dtype}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="google_robot_pick_coke_can")
    ap.add_argument("--steps", type=int, default=15)
    args = ap.parse_args()

    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    from scipy.spatial.transform import Rotation as R

    # inline copies of the harness helpers (kept in sync with scripts/simpler/main.py) so this smoke test is
    # self-contained and needs no openpi_client / tyro / imageio.
    def simpler_obs_to_state(obs):
        extra = obs.get("extra", {}) if isinstance(obs, dict) else {}
        tcp = np.asarray(extra.get("tcp_pose", np.zeros(7)), dtype=np.float32)
        pos = tcp[:3]
        if tcp.shape[0] >= 7:
            q = tcp[3:7]
            euler = R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz").astype(np.float32)
        else:
            euler = np.zeros(3, np.float32)
        agent = obs.get("agent", {}) if isinstance(obs, dict) else {}
        qpos = np.asarray(agent.get("qpos", []), dtype=np.float32)
        gripper = np.array([np.clip(qpos[-1], 0.0, 1.0)] if qpos.size else [0.0], np.float32)
        return np.concatenate([pos, euler, gripper]).astype(np.float32)

    def rt1_action_to_simpler(action):
        a = np.asarray(action, dtype=np.float32).copy()
        a[6] = 1.0 - 2.0 * np.clip(a[6], 0.0, 1.0)
        return a

    print(f"[smoke] make({args.task})")
    env = simpler_env.make(args.task)
    obs, reset_info = env.reset()
    instruction = env.get_language_instruction()
    print(f"[smoke] instruction: {instruction!r}")
    print(f"[smoke] action_space: {env.action_space}")
    print("[smoke] ===== obs structure (validate simpler_obs_to_state against this) =====")
    pretty(obs)
    print(f"[smoke] reset_info keys: {list(reset_info.keys()) if isinstance(reset_info, dict) else reset_info}")

    img = get_image_from_maniskill2_obs_dict(env, obs)
    print(f"[smoke] image: shape={np.asarray(img).shape} dtype={np.asarray(img).dtype} "
          f"min={np.asarray(img).min()} max={np.asarray(img).max()}")

    state = simpler_obs_to_state(obs)
    print(f"[smoke] simpler_obs_to_state -> shape={state.shape} value={np.round(state, 3)}")

    ok = True
    for t in range(args.steps):
        rt1_action = np.concatenate([np.random.uniform(-0.02, 0.02, 6), [np.random.choice([0.0, 1.0])]]).astype(np.float32)
        env_action = rt1_action_to_simpler(rt1_action)
        try:
            obs, reward, done, truncated, info = env.step(env_action)
        except Exception as e:
            print(f"[smoke] STEP FAILED at t={t}: {e}")
            ok = False
            break
        if t == 0:
            print(f"[smoke] step ok. info keys: {list(info.keys())}  reward={reward} done={done} truncated={truncated}")
            print(f"[smoke] 'success' in info: {'success' in info}  value={info.get('success')}")
    if ok:
        print(f"[smoke] ran {args.steps} random steps without error. env loop + helpers OK.")
        print("[smoke] NEXT: check the obs structure above matches simpler_obs_to_state's assumed keys "
              "(extra/tcp_pose, agent/qpos) and the image key; fix main.py if not.")


if __name__ == "__main__":
    main()
