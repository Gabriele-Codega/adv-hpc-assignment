#!/bin/bash
#SBATCH --job-name=matmul
#SBATCH -A ICT24_DSSC_GPU
#SBATCH --partition=boost_usr_prod

#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4

#SBATCH --time=00:10:00

## When sbatching, needs the argument --nodes=<N>
## Also needs one argument: size of the matrix


module load nvhpc/23.11 openmpi/4.1.6--nvhpc--23.11 cuda/

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=8
export OMP_PLACES=cores 
export OMP_PROC_BIND=true

if ! [ -e cuda_"$1"_times.csv ]; then
    touch cuda_"$1"_times.csv
    echo "npes, init, init_sd, comm, comm_sd, comp, comp_sd, copy, copy_sd" >> cuda_"$1"_times.csv
fi

srun --cpu-bind=verbose ./cuda.x "$1"
