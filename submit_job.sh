#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --mem-per-cpu=1.5G      # increase as needed
#SBATCH --time=1:00:00

module load python/3.9.6

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r MLRC2021-SimpleCopyPaste/requirements.txt

cp cityscapes.tar.gz $SLURM_TMPDIR/
tar -xzf $SLURM_TMPDIR/cityscapes.tar.gz
