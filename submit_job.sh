#!/bin/bash
#SBATCH --account=def-mjshafie
#SBATCH --nodes 1
#SBATCH --gres=gpu:p100:1 # request a GPU
#SBATCH --cpus-per-task=24
#SBATCH --mem=12000M
#SBATCH --time=2:0:0
#SBATCH --mail-user=smnair@uwaterloo.ca
#SBATCH --mail-type=ALL

tmux new-session -d -s "snair-job" './run_trainval.sh 2>&1 | tee /home/snair/scratch/tmp/tmux.log'

