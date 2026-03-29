#!/bin/bash
#SBATCH -JSlurmPythonExample                    # Job name
#SBATCH --account=gts-agarg35                   # charge account
#SBATCH -N1 --gres=gpu:RTX_6000:1               # Number of nodes and GPUs (change to desired count)
#SBATCH --mem-per-gpu=24G                      # Memory per core
#SBATCH -t8:00:00                               # Duration of the job (2 hours)
#SBATCH -q embers                               # QOS Name
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --output=slurm_out/Report-%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vgiridhar6@gatech.edu        # E-mail address for notifications

module load anaconda3/2022.05.0.1               # Load module dependencies
conda activate /storage/project/r-agarg35-0/vgiridhar6/.conda/envs/newt

echo "Running the following command:"
echo $@

export TMPDIR=/storage/home/hcoda1/6/vgiridhar6/wandb_tmp
export PYTHONWARNINGS="ignore::UserWarning:torchvision.io._video_deprecation_warning"
export MUJOCO_GL=egl
# Use accelerate launch for distributed training
# This will automatically set up DDP across all allocated GPUs
# Note: We skip the first argument ($1) which is "lerobot-train" since we specify it explicitly
# Run on single GPU
srun "$@"