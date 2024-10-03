#!/bin/bash
#SBATCH --job-name=jacobi
#SBATCH -A ICT24_DSSC_CPU
#SBATCH --partition=dcgp_usr_prod

#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=56

#SBATCH --mem=50000
#SBATCH --time=00:05:00

## When sbatching, needs the argument --nodes=<N>
## Also needs one argument: 
##     1. size of the matrix
module load intel-oneapi-compilers/ intel-oneapi-mpi/

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=56
export OMP_PLACES=cores 
export OMP_PROC_BIND=true

if ! [ -e cpu_"$1"_times.csv ]; then
    touch cpu_"$1"_times.csv
    echo "npes, init, init_sd, comm, comm_sd, comp, comp_sd, io, io_sd" >> cpu_"$1"_times.csv
fi

srun ./cpu.x "$1" 10 1 1
