#!/bin/bash
#SBATCH --job-name=compile
#SBATCH -A ICT24_DSSC_GPU
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=0
#SBATCH --mem=1000
#SBATCH --time=00:01:00

module load nvhpc/23.11 openmpi/4.1.6--nvhpc--23.11 cuda/

make gpu
make rma
