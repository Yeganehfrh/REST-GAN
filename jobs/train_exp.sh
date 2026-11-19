#!/bin/sh

#SBATCH --job-name=train_exp
#SBATCH --chdir=/home/users/mansarinia/EEGModalNet_exp/
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/users/mansarinia/EEGModalNet_exp/logs/train_exp_%j.log
#SBATCH --error=/home/users/mansarinia/EEGModalNet_exp/logs/train_exp_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y.farahzadi@gmail.com

# Exploratory runs from a separate checkout (git worktree EEGModalNet_exp), with the environment and the data
# of the main checkout. All options are passed to pipeline/train_gan_con.py.

echo "commit: $(git rev-parse HEAD)  branch: $(git rev-parse --abbrev-ref HEAD)"
git status --short --untracked-files=no

# SECTION Run pipeline
pixi run --manifest-path /home/users/mansarinia/EEGModalNet/pyproject.toml python -m src.EEGModalNet.pipeline.train_gan_con "$@"
