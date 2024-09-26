#!/bin/bash
#SBATCH --job-name=compile
#SBATCH -A itc24_dssc_cpu
#SBATCH --partition=dcgp_usr_prpod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000
#SBATCH --time=00:01:00

module load intel-oneapi-compilers/ intel-oneapi-mpi/ intel-oneapi-mkl/

make -f MakefileCPU
