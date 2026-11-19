#!/bin/sh

#SBATCH --job-name=train_gan_r2
#SBATCH --chdir=/home/users/mansarinia/EEGModalNet/
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=23:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/users/mansarinia/EEGModalNet/logs/train_gan_r2_%j.log
#SBATCH --error=/home/users/mansarinia/EEGModalNet/logs/train_gan_r2_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y.farahzadi@gmail.com

# Train REST-GAN. All options are passed to pipeline/train_gan_con.py (its defaults are the model of the paper), e.g.
#   sbatch jobs/train_gan_r2.sh --seed 1 --run-name main_s1 --heldout-split splits/lemon_heldout40.json
# The code is not pulled here: check out the intended commit before submitting.

echo "commit: $(git rev-parse HEAD)  branch: $(git rev-parse --abbrev-ref HEAD)"
git status --short --untracked-files=no

# SECTION Run pipeline
pixi run python -m src.EEGModalNet.pipeline.train_gan_con "$@"
