# Advanced HPC final project
This repo has the solutions to exercises assigned for the final exam of Advanced HPC at University of Trieste.

The exercises are the implementation of distributed matrix multiplication and the solution of the Laplace equation via Jacobi iterations. More details about the assignment can be found [here](https://github.com/Foundations-of-HPC/Advanced-High-Performance-Computing-2023/tree/main).

## Contents
- `matmul`: code and scripts for matrix multiplication
	- `main.c` is the main source file, which can be built to implement a naive algorithm, blas dgemm or cublas dgemm
	- `utils.*` declaration and implementation of utility functions
	- `*.sh` scripts use to submit jobs. They include scripts to compile and run the code. **Note**: all code was run on Leonardo at CINECA and the scripts are written accordingly
	- `Makefile*` makefiles for the different versions (CPU or GPU)
- `jacobi`: code and scripts for the Jacobi method
	- `jacobi_base.c` is the base serial version of the code, used mainly to validate the parallel implementations
	- `jacobi_parallel.c` is the parallel version, which can be compiled with both OpenMP and OpenACC
	- `jacobi_rma.c` is the same parallel code, except MPI communications use Remote Memory Access
	- `utils.*` declaration and implementation of utility functions 
	- `*.sh` scripts use to submit jobs. They include scripts to compile and run the code. **Note**: all code was run on Leonardo at CINECA and the scripts are written accordingly
	- `Makefile` makefile for all the versions
	- `plot.plt` script to plot the solution with Gnuplot
- `parallel_programming.pdf` is a short report with some information about the implementations and scalability results
