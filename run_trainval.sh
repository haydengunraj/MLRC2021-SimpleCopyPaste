#!/bin/bash

# Exit if any command fails
set -e

export PROJECT_HOME=/home/snair/projects/def-mjshafie/snair/syde671

module load python/3.9.6

cd $SLURM_TMPDIR

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

echo "Copying dataset"
cp $PROJECT_HOME/cityscapes.zip $SLURM_TMPDIR/

echo "Extracting dataset"
unzip $SLURM_TMPDIR/cityscapes.zip

echo "Copying repo"
cp -rf $PROJECT_HOME/MLRC2021-SimpleCopyPaste/ $SLURM_TMPDIR/
cd $SLURM_TMPDIR/MLRC2021-SimpleCopyPaste/

echo "Installing requirements.txt"
pip install --no-index -r cedar_requirements.txt

echo "Running trainval"
python trainval.py snair-test -rt $SLURM_TMPDIR/cityscapes --num_workers 16 --batch_size 8 --enable_wandb

