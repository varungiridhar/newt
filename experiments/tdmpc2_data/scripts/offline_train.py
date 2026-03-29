"""
Offline-only training of TDMPC2 agent on walker-run using MT80 demos.
No online environment interaction except periodic evaluation.

Usage:
    sbatch compute.sh python experiments/tdmpc2_data/scripts/offline_train.py
"""
import os
import sys
import time
import argparse
import numpy as np
import torch

os.environ['PYTHONUNBUFFERED'] = '1'

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
TDMPC2_DIR = os.path.join(REPO_ROOT, 'tdmpc2')
sys.path.insert(0, TDMPC2_DIR)
os.chdir(TDMPC2_DIR)
os.environ['MUJOCO_GL'] = 'egl'

# Monkey-patch hydra for parse_cfg (needs Hydra context for get_original_cwd)
import hydra.utils
hydra.utils.get_original_cwd = lambda: os.getcwd()

from omegaconf import OmegaConf
from config import Config, parse_cfg
from common.world_model import WorldModel
from tdmpc2 import TDMPC2
from common.buffer import Buffer
from envs import make_env
from common import set_seed
import wandb


def evaluate(agent, env, cfg, num_episodes=10):
    """Run evaluation episodes sequentially. Returns list of episode rewards."""
    agent.model.eval()
    rewards = []
    task = torch.zeros(cfg.num_envs, dtype=torch.long)

    for _ in range(num_episodes):
        obs, _ = env.reset()
        t0 = torch.ones(cfg.num_envs, dtype=torch.bool)
        ep_reward = 0.0
        done = False

        while not done:
            with torch.no_grad():
                action = agent(obs, t0=t0, eval_mode=True, task=task)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward.sum().item()
            done = (terminated | truncated).any().item()
            t0 = torch.zeros(cfg.num_envs, dtype=torch.bool)

        rewards.append(ep_reward)

    agent.model.train()
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='walker-run')
    parser.add_argument('--model_size', type=str, default='B')
    parser.add_argument('--total_steps', type=int, default=500_000)
    parser.add_argument('--eval_freq', type=int, default=25_000)
    parser.add_argument('--eval_episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(REPO_ROOT, 'experiments', 'tdmpc2_data'))
    args = parser.parse_args()

    # ── Config ─────────────────────────────────────────────────────────
    cfg = OmegaConf.structured(Config)
    cfg.task = args.task
    cfg.model_size = args.model_size
    cfg.seed = args.seed
    cfg.compile = False
    cfg.num_envs = 1  # Only used for eval
    cfg.data_dir = args.data_dir
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    exp_name = f'offline_{args.task}_{args.model_size}'
    print(f"\n{'='*60}")
    print(f"Offline TDMPC2: {args.task} (model={args.model_size})")
    print(f"Steps: {args.total_steps}, Eval every: {args.eval_freq}")
    print(f"{'='*60}")

    # ── Environment (eval only, also sets cfg.obs_shape/action_dim) ────
    print("\nCreating eval environment...")
    env = make_env(cfg)
    print(f"  obs_shape={cfg.obs_shape}, action_dim={cfg.action_dim}")

    # ── Agent ──────────────────────────────────────────────────────────
    print("Creating agent...", flush=True)
    torch.set_default_device(f'cuda:{cfg.rank}')
    model = WorldModel(cfg)
    agent = TDMPC2(model, cfg)
    torch.set_default_device('cpu')  # Reset for buffer/env operations
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ── Buffer + demos ─────────────────────────────────────────────────
    print("Loading demo data...")
    demo_path = os.path.join(args.data_dir, f'{args.task}.pt')
    assert os.path.exists(demo_path), f"Demo file not found: {demo_path}"
    td = torch.load(demo_path, weights_only=False)
    td['task'] = torch.full_like(td['reward'], 0, dtype=torch.int32)
    num_demo_steps = td['obs'].shape[0]
    num_demo_eps = td['episode'].max().item() + 1
    print(f"  {num_demo_steps} steps, {num_demo_eps} episodes")

    buffer = Buffer(
        capacity=max(num_demo_steps * 2, 200_000),
        batch_size=cfg.batch_size,
        horizon=cfg.horizon,
        cache_values=True,
        compile=False,
    )
    buffer.load_demos(td)
    del td

    # ── Wandb ──────────────────────────────────────────────────────────
    wandb.init(
        project='world_model_metric_analysis',
        entity='pair-diffusion',
        name=exp_name,
        config={
            'task': args.task, 'model_size': args.model_size,
            'total_steps': args.total_steps, 'eval_freq': args.eval_freq,
            'batch_size': cfg.batch_size, 'horizon': cfg.horizon,
            'seed': args.seed, 'mode': 'offline',
            'num_demo_steps': num_demo_steps, 'num_demo_eps': num_demo_eps,
        },
        job_type='offline_train',
    )

    # ── Initial eval ───────────────────────────────────────────────────
    print("\nInitial eval (untrained agent)...")
    eval_rewards = evaluate(agent, env, cfg, args.eval_episodes)
    print(f"  Reward: {np.mean(eval_rewards):.1f} +/- {np.std(eval_rewards):.1f}")
    wandb.log({'eval/reward_mean': np.mean(eval_rewards),
               'eval/reward_std': np.std(eval_rewards)}, step=0)

    # ── Offline training loop ──────────────────────────────────────────
    print(f"\nStarting offline training for {args.total_steps} steps...")
    log_freq = 1000
    metrics_accum = {}
    t_start = time.time()

    for step in range(1, args.total_steps + 1):
        train_metrics = agent.update(buffer)

        for k, v in train_metrics.items():
            metrics_accum.setdefault(k, []).append(
                v.item() if hasattr(v, 'item') else float(v))

        # Log training metrics
        if step % log_freq == 0:
            avg = {f'train/{k}': np.mean(v) for k, v in metrics_accum.items()}
            elapsed = time.time() - t_start
            avg['train/steps_per_sec'] = step / elapsed
            wandb.log(avg, step=step)
            loss_str = ', '.join(f'{k.replace("train/","")}={v:.4f}'
                                for k, v in sorted(avg.items())
                                if any(x in k for x in ['loss', 'consistency']))
            print(f"[{step:>7d}/{args.total_steps}] "
                  f"{avg['train/steps_per_sec']:.1f} it/s | {loss_str}")
            metrics_accum = {}

        # Evaluate
        if step % args.eval_freq == 0:
            eval_rewards = evaluate(agent, env, cfg, args.eval_episodes)
            em = {'eval/reward_mean': np.mean(eval_rewards),
                  'eval/reward_std': np.std(eval_rewards),
                  'eval/reward_min': np.min(eval_rewards),
                  'eval/reward_max': np.max(eval_rewards)}
            wandb.log(em, step=step)
            print(f"  EVAL @ {step}: {em['eval/reward_mean']:.1f} "
                  f"+/- {em['eval/reward_std']:.1f} "
                  f"[{em['eval/reward_min']:.1f}, {em['eval/reward_max']:.1f}]")

    # ── Final ──────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/3600:.1f}h")
    final_rewards = evaluate(agent, env, cfg, num_episodes=20)
    wandb.log({'eval/final_reward_mean': np.mean(final_rewards),
               'eval/final_reward_std': np.std(final_rewards)}, step=args.total_steps)
    print(f"Final: {np.mean(final_rewards):.1f} +/- {np.std(final_rewards):.1f}")

    ckpt_dir = os.path.join(REPO_ROOT, 'experiments', 'tdmpc2_data', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    agent.save(os.path.join(ckpt_dir, f'{exp_name}.pt'))
    print(f"Checkpoint saved to {ckpt_dir}/{exp_name}.pt")

    wandb.finish()
    env.close()


if __name__ == '__main__':
    main()
