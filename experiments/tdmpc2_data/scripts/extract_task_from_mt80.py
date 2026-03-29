"""
Extract single-task demo data from TD-MPC2 MT80 chunks.

Usage:
    python experiments/tdmpc2_data/scripts/extract_task_from_mt80.py --task walker-run

MT80 chunk data format: TensorDict with shape [num_episodes, 101, dim].
Obs padded to 39, action to 6 in MT80. Newt expects obs=128, action=16.
Tasks are scattered across chunks (not sequential).
"""
import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'tdmpc2'))
os.environ['MUJOCO_GL'] = 'egl'

import torch
from tensordict import TensorDict
from common import TASK_SET

# MT80 task_idx → chunk file mapping (from scanning all 20 chunks)
TASK_IDX_TO_CHUNK = {
    0: 18, 1: 19, 2: 17, 3: 5, 4: 14, 5: 15, 6: 0, 7: 14,
    8: 1, 9: 1, 10: 2, 11: 2, 12: 6, 13: 7, 14: 8, 15: 8,
    16: 9, 17: 11, 18: 10, 19: 18, 20: 16, 21: 4, 22: 5,
    23: 4, 24: 3, 25: 9, 26: 15, 27: 16, 28: 6, 29: 13,
    30: 11, 31: 11, 32: 11, 33: 11, 34: 11, 35: 11, 36: 11,
    37: 11, 38: 11, 39: 11, 40: 11, 41: 11, 42: 11, 43: 11,
    44: 11, 45: 12, 46: 11, 47: 12, 48: 12, 49: 12, 50: 12,
    51: 12, 52: 12, 53: 12, 54: 12, 55: 12, 56: 12, 57: 12,
    58: 12, 59: 12, 60: 12, 61: 12, 62: 12, 63: 12, 64: 12,
    65: 12, 66: 12, 67: 12, 68: 12, 69: 13, 70: 12, 71: 13,
    72: 13, 73: 13, 74: 13, 75: 11, 76: 11, 77: 11, 78: 11,
    79: 12,
}


def get_task_idx(task_name):
    """Get the MT80 task index for a given newt task name."""
    soup_tasks = TASK_SET['soup']
    if task_name in soup_tasks:
        idx = soup_tasks.index(task_name)
        if idx in TASK_IDX_TO_CHUNK:
            return idx
    # Try naming variants
    for variant in [task_name.replace('-backward', '-backwards'),
                    task_name.replace('-backwards', '-backward')]:
        if variant in soup_tasks:
            idx = soup_tasks.index(variant)
            if idx in TASK_IDX_TO_CHUNK:
                return idx
    raise ValueError(f"Task '{task_name}' not found in MT80 data.")


def get_true_dims(task_name):
    """Create the environment to get ground truth obs/action dims."""
    from types import SimpleNamespace
    from envs import (make_dm_control_env, make_metaworld_env, make_mujoco_env,
                      make_box2d_env, make_maniskill_env, make_robodesk_env,
                      make_ogbench_env, make_pygame_env, make_atari_env)
    cfg = SimpleNamespace(task=task_name, obs='state', seed=1, rank=0,
                          save_video=False, child_env=True, num_envs=1)
    for fn in [make_dm_control_env, make_metaworld_env, make_mujoco_env,
               make_box2d_env, make_maniskill_env, make_robodesk_env,
               make_ogbench_env, make_pygame_env, make_atari_env]:
        try:
            env = fn(cfg)
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            env.close()
            return obs_dim, action_dim
        except (ValueError, Exception):
            continue
    raise RuntimeError(f"Could not create environment for task '{task_name}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--max_episodes', type=int, default=1000,
                        help='Max episodes to extract (0=all)')
    parser.add_argument('--data_dir', type=str,
                        default='/storage/home/hcoda1/6/vgiridhar6/shared/tdmpc2_data/mt80')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(REPO_ROOT, 'experiments', 'tdmpc2_data'))
    args = parser.parse_args()

    # ── Step 1: Resolve task index and chunk ───────────────────────────────
    task_idx = get_task_idx(args.task)
    chunk_id = TASK_IDX_TO_CHUNK[task_idx]
    chunk_path = os.path.join(args.data_dir, f'chunk_{chunk_id}.pt')
    print(f"Task '{args.task}' -> task_idx={task_idx}, chunk_{chunk_id}.pt")
    assert os.path.exists(chunk_path), f"Chunk file not found: {chunk_path}"

    # ── Step 2: Get ground truth dims from environment ─────────────────────
    print(f"Creating '{args.task}' environment for ground truth dims...")
    true_obs_dim, true_action_dim = get_true_dims(args.task)
    print(f"Ground truth: obs_dim={true_obs_dim}, action_dim={true_action_dim}")

    # ── Step 3: Load chunk ─────────────────────────────────────────────────
    print(f"\nLoading {chunk_path} ...")
    chunk = torch.load(chunk_path, weights_only=False, map_location="cpu")
    for k in chunk.keys():
        print(f"  {k}: shape={chunk[k].shape}, dtype={chunk[k].dtype}")

    obs_data = chunk['obs']
    action_data = chunk['action']
    reward_data = chunk['reward']
    task_data = chunk['task']

    task_per_ep = task_data[:, 0]
    print(f"\nUnique tasks in chunk: {torch.unique(task_per_ep).tolist()}")

    # ── Step 4: Filter target task episodes ────────────────────────────────
    mask = task_per_ep == task_idx
    n_eps = mask.sum().item()
    assert n_eps > 0, f"No episodes found for task index {task_idx}!"
    print(f"'{args.task}' episodes available: {n_eps}")

    wr_obs = obs_data[mask]
    wr_action = action_data[mask]
    wr_reward = reward_data[mask]

    # Cap episode count
    if args.max_episodes > 0 and n_eps > args.max_episodes:
        print(f"Capping from {n_eps} to {args.max_episodes} episodes")
        n_eps = args.max_episodes
        wr_obs = wr_obs[:n_eps]
        wr_action = wr_action[:n_eps]
        wr_reward = wr_reward[:n_eps]

    # Free chunk memory
    del chunk, obs_data, action_data, reward_data, task_data

    # ── Step 5: Re-pad to newt's expected dims ─────────────────────────────
    from envs.wrappers.vectorized_multitask import MAX_OBS_DIM, MAX_ACTION_DIM

    mt80_obs_dim = wr_obs.shape[-1]
    mt80_action_dim = wr_action.shape[-1]
    print(f"\nMT80 padded dims: obs={mt80_obs_dim}, action={mt80_action_dim}")
    print(f"Newt target dims: obs={MAX_OBS_DIM}, action={MAX_ACTION_DIM}")

    # Trim MT80 padding to true dims, then re-pad to newt dims
    wr_obs = wr_obs[:, :, :true_obs_dim]
    wr_action = wr_action[:, :, :true_action_dim]

    if true_obs_dim < MAX_OBS_DIM:
        obs_pad = torch.zeros(*wr_obs.shape[:2], MAX_OBS_DIM - true_obs_dim)
        wr_obs = torch.cat([wr_obs, obs_pad], dim=-1)

    if true_action_dim < MAX_ACTION_DIM:
        act_pad = torch.zeros(*wr_action.shape[:2], MAX_ACTION_DIM - true_action_dim)
        wr_action = torch.cat([wr_action, act_pad], dim=-1)

    print(f"Final dims: obs={wr_obs.shape[-1]}, action={wr_action.shape[-1]}")

    # ── Step 6: Flatten to newt demo format ────────────────────────────────
    ep_len = wr_obs.shape[1]
    total_steps = n_eps * ep_len

    flat_obs = wr_obs.reshape(total_steps, MAX_OBS_DIM)
    flat_action = wr_action.reshape(total_steps, MAX_ACTION_DIM)
    flat_reward = wr_reward.reshape(total_steps)
    flat_episode = torch.arange(n_eps, dtype=torch.int64).unsqueeze(1).expand(n_eps, ep_len).reshape(total_steps)

    print(f"\nFlattened: {total_steps} steps, {n_eps} episodes of length {ep_len}")

    td = TensorDict({
        'obs': flat_obs,
        'action': flat_action,
        'reward': flat_reward,
        'episode': flat_episode,
    }, batch_size=(total_steps,))

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.task}.pt")
    torch.save(td, output_path)
    file_size = os.path.getsize(output_path) / 1e6
    print(f"\nSaved to {output_path} ({file_size:.1f} MB)")

    # ── Step 7: Verification ──────────────────────────────────────────────
    td_check = torch.load(output_path, weights_only=False)
    print(f"\nVerification:")
    for k in td_check.keys():
        print(f"  {k}: {td_check[k].shape}, dtype={td_check[k].dtype}")
    print(f"  num_episodes: {td_check['episode'].max().item() + 1}")

    for ep_id in range(min(5, n_eps)):
        ep_mask = flat_episode == ep_id
        r = flat_reward[ep_mask]
        valid_r = r[~torch.isnan(r)]
        print(f"  Episode {ep_id}: len={ep_mask.sum().item()}, "
              f"reward_sum={valid_r.sum().item():.2f}, nan_count={torch.isnan(r).sum().item()}")


if __name__ == '__main__':
    main()
