# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a Python/PyTorch RL research codebase (**MMBench & Newt**) for training massively multitask world models. The main entry point is `tdmpc2/train.py` (invoked via Hydra). There are no automated tests or linting configurations in the repository.

### Environment activation

Always activate the conda environment before running any Python commands:

```bash
eval "$(/opt/miniconda3/bin/conda shell.bash hook)" && conda activate newt
```

### Key environment variables

Set these before running any training or environment code:

- `MUJOCO_GL=osmesa` — required for headless MuJoCo rendering (no GPU display); alternatives are `egl` (GPU) or `glfw` (display)
- `MS_SKIP_ASSET_DOWNLOAD_PROMPT=1` — prevents interactive ManiSkill asset download prompts
- `SDL_VIDEODRIVER=dummy` — needed for pygame environments in headless mode

### GPU requirement

`train.py` has `assert torch.cuda.is_available()` at line 178. Training **requires** a CUDA GPU. Without a GPU, you can still:

- Import and test all modules
- Create and interact with RL environments (DMControl, MuJoCo, Box2D, Pygame, etc.)
- Instantiate and run WorldModel forward passes on CPU

### Running the application

See `README.md` for training commands. Example single-task training (requires GPU):

```bash
cd /workspace/tdmpc2 && python train.py task=walker-walk model_size=B
```

### Dependency note

`setuptools` must be pinned below version 81 (`setuptools<81`) because `sapien` (used by ManiSkill) depends on the deprecated `pkg_resources` module, which was removed in setuptools 82+.

### Project structure

- `tdmpc2/` — main source code (config, environments, world model, agent, trainer)
- `docker/` — Dockerfile and `environment.yaml` (conda deps)
- `tasks.json` — precomputed task embeddings and metadata (required at runtime)
- `csv/` — precomputed benchmark results
