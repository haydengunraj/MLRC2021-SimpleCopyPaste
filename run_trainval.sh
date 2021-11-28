#!/bin/bash

module load python/3.9.6

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

cd /home/snair/projects/def-mjshafie/snair/syde671

echo "Copying dataset"
cp cityscapes.tar.gz $SLURM_TMPDIR/

echo "Extracting dataset"
tar -xvzf $SLURM_TMPDIR/cityscapes.tar.gz

echo "Copying repo"
cp -rf MLRC2021-SimpleCopyPaste/ $SLURM_TMPDIR/
cd $SLURM_TMPDIR/MLRC2021-SimpleCopyPaste/

echo "Installing requirements.txt"
pip install --no-index -r cedar_requirements.txt

echo "Running trainval"
python trainval.py snair-test -rt $SLURM_TMPDIR/cityscapes --num_workers 22

