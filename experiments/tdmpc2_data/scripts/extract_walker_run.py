"""
Extract walker-run demo data from TD-MPC2 MT80 chunks.

MT80 task ordering: walker-run is index 2, in chunk_0 (tasks 0-3).
Chunk format: [num_episodes, 101, dim] with obs padded to 39, action dim 6.
walker-run true dims: obs=24, action=6.
"""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tdmpc2'))
os.environ['MUJOCO_GL'] = 'egl'

CHUNK_PATH = "/storage/home/hcoda1/6/vgiridhar6/shared/tdmpc2_data/mt80/chunk_0.pt"
WALKER_RUN_TASK_IDX = 2
OUTPUT_DIR = "/storage/home/hcoda1/6/vgiridhar6/forks/newt/experiments/tdmpc2_data"

# ── Step 1: Get ground truth dims from the environment ─────────────────────
print("Creating walker-run environment to get ground truth dims...")
from config import Config, parse_cfg
from omegaconf import OmegaConf

cfg = OmegaConf.structured(Config)
cfg.task = "walker-run"
cfg.num_envs = 1
cfg.obs = "state"
cfg = parse_cfg(cfg)

from envs import make_env
env = make_env(cfg)
obs, info = env.reset()
rand_action = env.rand_act()
true_obs_dim = obs.shape[-1]
true_action_dim = rand_action.shape[-1]
env.close()
print(f"Ground truth: obs_dim={true_obs_dim}, action_dim={true_action_dim}")

# ── Step 2: Load and inspect chunk ─────────────────────────────────────────
print("\nLoading chunk_0.pt ...")
chunk = torch.load(CHUNK_PATH, weights_only=False, map_location="cpu")

for k in chunk.keys():
    v = chunk[k]
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

obs_data = chunk['obs']       # [N_eps, 101, 39]
action_data = chunk['action'] # [N_eps, 101, 6]
reward_data = chunk['reward'] # [N_eps, 101]
task_data = chunk['task']     # [N_eps, 101]

# Task ID is constant within each episode; use first timestep
task_per_ep = task_data[:, 0]  # [N_eps]
print(f"\nUnique tasks in chunk: {torch.unique(task_per_ep).tolist()}")
for t in torch.unique(task_per_ep):
    count = (task_per_ep == t).sum().item()
    print(f"  Task {t.item()}: {count} episodes")

# ── Step 3: Filter walker-run episodes ─────────────────────────────────────
mask = task_per_ep == WALKER_RUN_TASK_IDX
n_eps = mask.sum().item()
print(f"\nWalker-run episodes: {n_eps}")

wr_obs = obs_data[mask]       # [n_eps, 101, 39]
wr_action = action_data[mask] # [n_eps, 101, 6]
wr_reward = reward_data[mask] # [n_eps, 101]

# ── Step 4: Trim obs padding ──────────────────────────────────────────────
padded_obs_dim = wr_obs.shape[-1]
padded_action_dim = wr_action.shape[-1]
print(f"Padded dims: obs={padded_obs_dim}, action={padded_action_dim}")
print(f"True dims:   obs={true_obs_dim}, action={true_action_dim}")

if padded_obs_dim > true_obs_dim:
    padding = wr_obs[:, :, true_obs_dim:]
    print(f"Obs padding stats: mean={padding.mean():.6f}, max_abs={padding.abs().max():.6f}, "
          f"nonzero={(padding != 0).sum().item()}")
    wr_obs = wr_obs[:, :, :true_obs_dim]

if padded_action_dim > true_action_dim:
    padding = wr_action[:, :, true_action_dim:]
    print(f"Action padding stats: mean={padding.mean():.6f}, max_abs={padding.abs().max():.6f}")
    wr_action = wr_action[:, :, :true_action_dim]

# ── Step 5: Flatten to newt demo format ────────────────────────────────────
# newt load_demos expects flat TensorDict: [total_steps] with obs, action, reward, episode
# Each episode is a contiguous block identified by the episode column

ep_len = wr_obs.shape[1]  # 101
total_steps = n_eps * ep_len

# Flatten: [n_eps, ep_len, dim] -> [n_eps * ep_len, dim]
flat_obs = wr_obs.reshape(total_steps, true_obs_dim)
flat_action = wr_action.reshape(total_steps, true_action_dim)
flat_reward = wr_reward.reshape(total_steps)

# Create episode IDs: 0,0,...,0, 1,1,...,1, ...
flat_episode = torch.arange(n_eps).unsqueeze(1).expand(n_eps, ep_len).reshape(total_steps).to(torch.int64)

print(f"\nFlattened shapes:")
print(f"  obs: {flat_obs.shape}")
print(f"  action: {flat_action.shape}")
print(f"  reward: {flat_reward.shape}")
print(f"  episode: {flat_episode.shape}")

# Save as TensorDict
from tensordict import TensorDict
td = TensorDict({
    'obs': flat_obs,
    'action': flat_action,
    'reward': flat_reward,
    'episode': flat_episode,
}, batch_size=(total_steps,))

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, "walker-run.pt")
torch.save(td, output_path)
print(f"\nSaved walker-run demos to {output_path}")

# ── Step 6: Verification ──────────────────────────────────────────────────
td_check = torch.load(output_path, weights_only=False)
print(f"\nVerification:")
for k in td_check.keys():
    print(f"  {k}: {td_check[k].shape}, dtype={td_check[k].dtype}")
print(f"  num_episodes: {td_check['episode'].max().item() + 1}")
print(f"  episode_length: {ep_len}")

# Print some episode stats
for ep_id in range(min(5, n_eps)):
    ep_mask = flat_episode == ep_id
    ep_rew = flat_reward[ep_mask].sum().item()
    ep_obs_range = flat_obs[ep_mask]
    print(f"  Episode {ep_id}: len={ep_mask.sum().item()}, total_reward={ep_rew:.2f}, "
          f"obs_mean={ep_obs_range.mean():.4f}, obs_std={ep_obs_range.std():.4f}")
