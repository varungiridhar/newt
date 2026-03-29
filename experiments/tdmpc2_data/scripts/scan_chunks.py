"""
Scan all MT80 chunks to build a task_idx → chunk mapping.
Prints which tasks are in which chunks.
"""
import os
import torch

DATA_DIR = "/storage/home/hcoda1/6/vgiridhar6/shared/tdmpc2_data/mt80"

chunk_files = sorted(
    [f for f in os.listdir(DATA_DIR) if f.startswith('chunk_') and f.endswith('.pt')],
    key=lambda x: int(x.split('_')[1].split('.')[0])
)

task_to_chunk = {}

for cf in chunk_files:
    path = os.path.join(DATA_DIR, cf)
    print(f"\nLoading {cf} ({os.path.getsize(path)/1e9:.1f} GB)...")
    chunk = torch.load(path, weights_only=False, map_location="cpu")

    task_data = chunk['task']
    obs_data = chunk['obs']
    action_data = chunk['action']

    task_per_ep = task_data[:, 0]
    unique_tasks = torch.unique(task_per_ep).tolist()

    print(f"  Shape: obs={obs_data.shape}, action={action_data.shape}, "
          f"reward={chunk['reward'].shape}")
    print(f"  Tasks: {[int(t) for t in unique_tasks]}")
    for t in unique_tasks:
        t = int(t)
        count = (task_per_ep == t).sum().item()
        print(f"    Task {t}: {count} episodes")
        task_to_chunk[t] = cf

    del chunk  # Free memory

print("\n\n=== FULL TASK → CHUNK MAPPING ===")
for task_idx in sorted(task_to_chunk.keys()):
    print(f"  Task {task_idx:3d} → {task_to_chunk[task_idx]}")
