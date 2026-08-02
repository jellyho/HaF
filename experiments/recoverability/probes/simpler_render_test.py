import os, numpy as np
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
env = simpler_env.make("google_robot_pick_coke_can")
obs, info = env.reset(seed=0)
img = get_image_from_maniskill2_obs_dict(env, obs)
instr = env.get_language_instruction()
# take one zero-ish step to confirm step() works
act = env.action_space.sample()*0.0
obs, rew, done, trunc, info = env.step(act)
print(f"RENDER_OK image={np.asarray(img).shape} dtype={np.asarray(img).dtype} instr={instr!r} success={info.get('success')}", flush=True)
