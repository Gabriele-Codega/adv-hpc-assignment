#include <stdio.h>
void init_mat(double* A, int size, int rank, int npes);
void eye(double* B, int size, int rank, int npes);
void print_mat(int N,int M,double* mat,FILE* file);
void print_mat_parallel(double* A, int rank, int size, int n_loc, int npes,FILE* file);
void create_block(const double* B, const int nBcols, const int start, const int nrows, const int ncols, double* block);
void matmul(const double* A, const double* B, const int N, const int K, const int M, double* C, const int n_Ccols, const int start);
