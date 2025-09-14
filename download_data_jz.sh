#!/bin/bash
#SBATCH --job-name=wget    # job name
#SBATCH --account=fbe@a100
#SBATCH --partition=archive
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --cpus-per-task=1            # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=05:59:59               # maximum execution time (HH:MM:SS)
#SBATCH --signal=SIGUSR1@90
#SBATCH --output=logs/wget/test%j.out # output file name
#SBATCH --error=logs/wget/test%j.err  # error file name

set -x
cd ${SLURM_SUBMIT_DIR}

module purge

cd /lustre/fsn1/projects/rech/fbe/commun/datasets/PoM/geo/

wget https://zenodo.org/records/5735646/files/PASTIS-R.zip?download=1
echo "Download finished"
unzip 'PASTIS-R.zip?download=1'
echo "Unzip finished"
mv 'PASTIS-R' PASTIS-R