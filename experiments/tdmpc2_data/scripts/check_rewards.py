"""Quick diagnostic on the extracted walker-run data to understand NaN rewards."""
import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'tdmpc2'))

import torch

# Check the extracted data
td = torch.load(os.path.join(REPO_ROOT, 'experiments/tdmpc2_data/walker-run.pt'),
                weights_only=False, map_location='cpu')
print("=== Extracted data ===")
for k in td.keys():
    v = td[k]
    nan_count = torch.isnan(v).sum().item() if v.is_floating_point() else 0
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}, nan_count={nan_count}, "
          f"min={v[~torch.isnan(v)].min().item() if nan_count < v.numel() else 'all NaN'}, "
          f"max={v[~torch.isnan(v)].max().item() if nan_count < v.numel() else 'all NaN'}")

reward = td['reward']
ep = td['episode']
# Check first few episodes in detail
for ep_id in range(3):
    mask = ep == ep_id
    r = reward[mask]
    nan_mask = torch.isnan(r)
    print(f"\nEpisode {ep_id} (len={mask.sum()}):")
    print(f"  NaN positions: {torch.where(nan_mask)[0].tolist()[:10]}{'...' if nan_mask.sum()>10 else ''}")
    print(f"  NaN count: {nan_mask.sum().item()}/{len(r)}")
    valid = r[~nan_mask]
    if len(valid) > 0:
        print(f"  Valid rewards: mean={valid.mean():.4f}, min={valid.min():.4f}, max={valid.max():.4f}")
        print(f"  First 10 values: {r[:10].tolist()}")
    else:
        print(f"  All NaN!")

# Also check the raw chunk data directly
print("\n=== Raw chunk_17 reward check ===")
chunk = torch.load('/storage/home/hcoda1/6/vgiridhar6/shared/tdmpc2_data/mt80/chunk_17.pt',
                   weights_only=False, map_location='cpu')
task_per_ep = chunk['task'][:, 0]
mask = task_per_ep == 2
raw_reward = chunk['reward'][mask]
print(f"Raw reward shape: {raw_reward.shape}")
print(f"NaN count: {torch.isnan(raw_reward).sum().item()}/{raw_reward.numel()}")
print(f"First episode rewards[:10]: {raw_reward[0, :10].tolist()}")
print(f"First episode rewards[-5:]: {raw_reward[0, -5:].tolist()}")
valid = raw_reward[~torch.isnan(raw_reward)]
if len(valid) > 0:
    print(f"Valid: mean={valid.mean():.4f}, min={valid.min():.4f}, max={valid.max():.4f}")

# Check obs for NaN too
raw_obs = chunk['obs'][mask]
print(f"\nRaw obs NaN count: {torch.isnan(raw_obs[:, :, :24]).sum().item()}/{raw_obs[:, :, :24].numel()}")
print(f"First ep, step 0, obs[:5]: {raw_obs[0, 0, :5].tolist()}")
print(f"First ep, step 100, obs[:5]: {raw_obs[0, 100, :5].tolist()}")

# Check action for NaN
raw_action = chunk['action'][mask]
print(f"\nRaw action NaN count: {torch.isnan(raw_action).sum().item()}/{raw_action.numel()}")
print(f"First ep, step 0, action: {raw_action[0, 0].tolist()}")
