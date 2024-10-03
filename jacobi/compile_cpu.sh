#!/bin/bash
#SBATCH --job-name=compile
#SBATCH -A ICT24_DSSC_CPU
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000
#SBATCH --time=00:01:00

module load intel-oneapi-compilers/ intel-oneapi-mpi/

make cpu
