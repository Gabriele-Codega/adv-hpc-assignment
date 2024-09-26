#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "utils.h"

void init_mat(double* A, int size, int rank, int npes){
    int rest = size%npes;
    int n_loc = size/npes + (rank < rest);
    int offset = rest * (rank >= rest);
    int ig;
    #pragma omp parallel for private(ig) 
    for(int i =0;i<n_loc;i++){
	for(int j = 0;j<size;j++){
	    ig = i + n_loc * rank + offset;
	    A[i*size+ j] = ig * size + j;
	}
    }
}

void eye(double* B, int size, int rank, int npes){
    int rest = size%npes;
    int n_loc = size/npes + (rank < rest);
    int offset = rest * (rank >= rest);
    int ig;
    #pragma omp parallel for private(ig)
    for(int i =0;i<n_loc;i++){
	for(int j = 0;j<size;j++){
	    ig = i + n_loc * rank + offset;
	    B[i*size + ig] = 1;
	}
    }
}

void print_mat(int N,int M,double* mat,FILE* file){
    int i = 0, j = 0;
    for (i=0;i<N;i++){
        for (j=0;j<M;j++){
            fprintf(file,"%.3g\t",mat[i*M+j]);
        }
        fprintf(file,"\n");
    }
}

void print_mat_parallel(double* A, int rank, int size, int n_loc, int npes,FILE* file){
    if(!rank){
        double* buf = (double*) malloc(size*n_loc*sizeof(double));
    	print_mat(n_loc,size,A,file);
        for (int count = 1;count < npes; count++){
            MPI_Status status;
            MPI_Recv(buf,size*n_loc,MPI_DOUBLE,count,100,MPI_COMM_WORLD,&status);
            int msg_size = 0;
            MPI_Get_count(&status,MPI_DOUBLE,&msg_size);
            print_mat(msg_size/size,size,buf,file);
        }
        fflush(file);
    }
    else MPI_Send(A,size*n_loc,MPI_DOUBLE,0,100,MPI_COMM_WORLD);
}

void create_block(const double* B, const int nBcols, const int start, const int nrows, const int ncols, double* block){
    #pragma omp parallel for
    for (int i= 0;i<nrows;i++){
	for(int j = 0; j<ncols;j++){
	   block[i*ncols+j] = B[i*nBcols + start + j]; 
	}
    }
}

void matmul(const double* A, const double* B, const int N, const int K, const int M, double* C, const int n_Ccols, const int start){
    /* computes the matrix product A*B = C, where A is of size NxK, B is of size KxM and C is of size NxM 
     * 
     * Parameters:
     * A: pointer to matrix A
     * B: pinter to matrix B
     * N: number of rows of A
     * K: number of columns of A and number of rows of B
     * M: number of columns of B
     * C: pointer to the matrix C
     * n_Ccols: number of columns of C, needed if computing a block of C instead of the whole thing
     * start: column of C where the block that is beaing computed starts
     *
     * */
    register int a_row,c_row;
    #pragma omp parallel for private(a_row,c_row)
    for (int i = 0;i<N;i++){
	for (int j = 0;j<M;j++){
	    a_row = i*K;
	    c_row = i*n_Ccols;
	    double tot = 0;
	    for (int k = 0;k<K;k++){
		tot += A[a_row + k] * B[k*M + j];
	    }
	    C[c_row + j + start] = tot;
	}
    }

}
