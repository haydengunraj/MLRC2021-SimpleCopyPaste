#!/bin/bash
#SBATCH --account=def-mjshafie
#SBATCH --nodes 1
#SBATCH --gres=gpu:p100:1 # request a GPU
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G      
#SBATCH --time=0-24:00


module load python/3.9.6

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

cd /home/snair/projects/def-mjshafie/snair/syde671

cp cityscapes.tar.gz $SLURM_TMPDIR/
tar -xzf $SLURM_TMPDIR/cityscapes.tar.gz

cp -rf MLRC2021-SimpleCopyPaste/ $SLURM_TMPDIR/
cd $SLURM_TMPDIR/MLRC2021-SimpleCopyPaste/

pip install --no-index -r requirements.txt

python trainval.py snair-test -rt $SLURM_TMPDIR/cityscapes --num_workers 22



