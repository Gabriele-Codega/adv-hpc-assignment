#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


#ifdef TIMEIT
#define timeit(x,time)\
    do { \
	double t1 = MPI_Wtime(); \
	x; \
	double t2 = MPI_Wtime(); \
	*(time) += t2-t1; \
    } while(0)
#else
#define timeit(x,time) do {x;} while(0)
#endif

//#define SIZE 2500

void init_mat(double* A, int size, int rank, int npes){
    int rest = size%npes;
    int n_loc = size/npes + (rank < rest);
    int offset = rest * (rank >= rest);
    int ig;
    #pragma omp parallel for private(ig) collapse(2)
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
    #pragma omp parallel for private(ig) collapse(2)
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
            fprintf(file,"%.3f\t",mat[i*M+j]);
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
    #pragma omp parallel for private(a_row,c_row) //collapse(2)
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


int main(int argc,char** argv){

    int SIZE = 10;
    if (argc > 1){
	SIZE = atoi(argv[1]);
    }

    // time variables
    double t_init = 0, t_comm = 0, t_compute = 0;

    int rank,npes;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);


    int rest = SIZE%npes;
    int n_loc = SIZE/npes + (rank < rest);
    int offset = rest * (rank >= rest);


    int* workloads = (int*) calloc(npes,sizeof(int));
    //MPI_Allgather(&n_loc, 1, MPI_INTEGER,workloads, 1, MPI_INTEGER, MPI_COMM_WORLD);
    for (int i=0; i<npes; i++){
	workloads[i] = SIZE/npes + (i < rest);
    }
    // assuming both matrices of size SIZExSIZE for now
    double *A = (double*) calloc(n_loc*SIZE,sizeof(double));
    double *B = (double*) calloc(n_loc*SIZE,sizeof(double));
    double *C = (double*) calloc(n_loc*SIZE,sizeof(double));

    // initialise to some data
    timeit({init_mat(A,SIZE,rank,npes); eye(B,SIZE,rank,npes);},&t_init);


    // print the matrix A before computation
    #ifdef DEBUG 
    print_mat_parallel(A,rank,SIZE,n_loc,npes,stdout);
    #endif

    // set up the counts and displacements fo allgatherv
    int* rcvcounts = (int*) malloc(npes*sizeof(int));
    // for the first count < rest communications, the number of columns is size/npes + 1, i.e. workloads[0]
    int ncols = workloads[0];
    for (int i = 0;i<npes;i++){
	rcvcounts[i] = workloads[i]*ncols;
    }
    int* displs = (int*) malloc(npes*sizeof(int));
    displs[0] = 0;
    for (int i=1;i<npes;i++){
	displs[i] = displs[i-1] + rcvcounts[i-1];
    }
    double* b_col = (double*) malloc(SIZE*ncols*sizeof(double));
    double* b_block = (double*) malloc(n_loc*ncols*sizeof(double));

    int col_idx = 0;
    for (int count = 0;count<npes;count++){
	// when count >= rest, the number of columns is set to sizd/npes.
	// arrays with receive counts and displacements are updated accordingly and 
	// buffers for blocks are reallocated
	if (count == rest){
	    ncols = workloads[count];
	    for (int i = 0;i<npes;i++){
		rcvcounts[i] = workloads[i]*ncols;
	    }
	    displs[0] = 0;
    	    for (int i=1;i<npes;i++){
    	        displs[i] = displs[i-1] + rcvcounts[i-1];
    	    }
	    b_col = (double*) realloc(b_col, SIZE*ncols*sizeof(double));
	    b_block = (double*) realloc(b_block, n_loc*ncols*sizeof(double));
	}


	// create block to send and gather
	timeit( { 
		 create_block(B,SIZE,col_idx,n_loc,ncols,b_block);
		 MPI_Allgatherv(b_block,n_loc*ncols,MPI_DOUBLE,b_col,rcvcounts,displs,MPI_DOUBLE,MPI_COMM_WORLD); 
		}, 
		&t_comm);

//	if(!rank){
//	    printf("------ %d ------\n",count);
//	    for (int i = 0;i<npes;i++){
//		printf("%d ",rcvcounts[i]);
//	    }
//	    printf("\n");
//	    for (int i = 0;i<npes;i++){
//		printf("%d ",displs[i]);
//	    }
//	    printf("\n");
//
//	    double* buf = (double*) malloc(SIZE*ncols*sizeof(double));
//	    print_mat(SIZE,ncols,b_col,stdout);
//	    printf("\n\n");
//	    for (int count = 1;count < npes; count++){
//		MPI_Status status;
//		MPI_Recv(buf,SIZE*ncols,MPI_DOUBLE,count,100,MPI_COMM_WORLD,&status);
//		int msg_size = 0;
//		MPI_Get_count(&status,MPI_DOUBLE,&msg_size);
//		print_mat(msg_size/ncols,ncols,buf,stdout);
//		printf("\n\n");
//	    }
//	}
//	else MPI_Send(b_col,SIZE*ncols,MPI_DOUBLE,0,100,MPI_COMM_WORLD);

//	printf("%d, %d, %d, %d\n",n_loc,SIZE,ncols,col_idx);
	timeit( matmul(A,b_col,n_loc,SIZE,ncols,C,SIZE,col_idx) , &t_compute); 
	col_idx += ncols;

    }
    free(b_block);
    free(b_col);

    // print result of the computation
    #ifdef DEBUG 
    print_mat_parallel(C,rank,SIZE,n_loc,npes,stdout);
    #endif

    free(A);
    free(B);
    free(C);

    double* times_init = NULL;
    double* times_comm = NULL;
    double* times_compute = NULL;
    if (!rank){
	times_init = (double*) malloc(npes*sizeof(double));
	times_comm = (double*) malloc(npes*sizeof(double));
	times_compute = (double*) malloc(npes*sizeof(double));
    }
    MPI_Gather(&t_init, 1, MPI_DOUBLE, times_init, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&t_comm, 1, MPI_DOUBLE, times_comm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&t_compute, 1, MPI_DOUBLE, times_compute, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (!rank){
	double init_mean = 0, comm_mean = 0, compute_mean = 0;
	for (int i = 0;i<npes;i++){
	    init_mean += times_init[i];
	    comm_mean += times_comm[i];
	    compute_mean += times_compute[i];
	}
	init_mean /= npes;
	comm_mean /= npes;
	compute_mean /= npes;

	double init_std = 0, comm_std = 0, compute_std = 0;
	for (int i = 0;i<npes;i++){
	    init_std += (times_init[i] - init_mean) * (times_init[i] - init_mean);
	    comm_std += (times_comm[i] - comm_mean) * (times_comm[i] - comm_mean);
	    compute_std += (times_compute[i] - compute_mean) * (times_compute[i] - compute_mean);
	}
	init_std /= (npes-1);
	comm_std /= (npes-1);
	compute_std /= (npes-1);

	init_std = sqrt(init_std);
	comm_std = sqrt(comm_std);
	compute_std = sqrt(compute_std);

	printf("Times:\n\tinit: %.3g s pm %3.g\n\tcomm: %.3g s pm %3.g\n\tcomputation: %.3g s pm %3.g\n",init_mean,init_std,comm_mean,comm_std,compute_mean,compute_std);
    }


    //printf("proc %d:\n\tinit: %.3g s\n\tcomm: %.3g s\n\tcomputation: %.3g s\n",rank,t_init,t_comm,t_compute);

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}
