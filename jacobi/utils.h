#include <stdio.h>
void evolve( double * matrix, double *matrix_new, size_t nloc, size_t dimension );
void save_gnuplot( double *M, size_t dimension);
void save_gnuplot_distributed(double* A, int rank, int size, int n_loc, int npes);
void print_mat(int N,int M,double* mat,FILE* file);
void print_mat_parallel(double* A, int rank, int size, int n_loc, int npes,FILE* file);
void save_gnuplot_mpi (double*matrix, int rank, int dimension, int n_loc, int npes, int rest);
