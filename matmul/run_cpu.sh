#!/bin/bash
#SBATCH --job-name=compile
#SBATCH -A ICT24_DSSC_CPU
#SBATCH --partition=dcgp_usr_prod

#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=56

#SBATCH --mem=50000
#SBATCH --time=01:00:00

## When sbatching, needs the argument --nodes=<N>
## Also needs two arguments: 
##     1. name of the executable, without extension ('naive' or 'blas')
##     2. size of the matrix
module load intel-oneapi-compilers/ intel-oneapi-mpi/ intel-oneapi-mkl/

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=56
export OMP_PLACES=cores 
export OMP_PROC_BIND=true

if ! [ -e "$1"_"$2"_times.csv ]; then
    touch "$1"_"$2"_times.csv
    echo "npes, init, init_sd, comm, comm_sd, comp, comp_sd, copy, copy_sd" >> "$1"_"$2"_times.csv
fi

srun ./"$1".x "$2"
