#!/bin/bash

for i in {1,2,4,8,16}; do 
    sbatch --nodes "$i" run_gpu.sh 5000
done
