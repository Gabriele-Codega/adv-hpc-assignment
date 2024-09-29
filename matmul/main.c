#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "utils.h"

#ifdef BLAS
#include <mkl.h>
#endif

#ifdef CUBLAS
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#endif

// macro to check cuda errors, yanked from some online post
#ifdef CUBLAS
#define cudaCheckError() { \
	    cudaError_t err=cudaGetLastError(); \
	    if (err != cudaSuccess){ \
		fprintf(stderr,"Cuda error %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err)); \
		exit(err); \
	    }\
	}    
#endif

// macros to time stuff
#ifdef TIMEIT
#define timeCPU(x,time)\
    do { \
	double t1 = MPI_Wtime(); \
	x; \
	double t2 = MPI_Wtime(); \
	*(time) += t2-t1; \
    } while(0)
#define timeGPU(x,time)\
    do { \
	cudaEvent_t start,stop; \
	cudaEventCreate(&start); \
	cudaEventCreate(&stop); \
	cudaEventRecord(start,0); \
	x; \
	cudaEventRecord(stop,0); \
	cudaEventSynchronize(stop); \
	float elapsed; \
	cudaEventElapsedTime(&elapsed,start,stop); \
	*(time) += elapsed/1000; \
    } while(0)
#else
#define timeCPU(x,time) do {x;} while(0)
#define timeGPU(x,time) do {x;} while(0)
#endif





int main(int argc,char** argv){

    int SIZE = 10;
    if (argc > 1){
	SIZE = atoi(argv[1]);
    }

    // time variables
    double t_init = 0, t_comm = 0, t_block = 0, t_gather = 0, t_compute = 0, t_copy = 0;

    int rank,npes;
    int th_level;
    MPI_Init_thread(NULL,NULL,MPI_THREAD_FUNNELED,&th_level);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    int rest = SIZE%npes;
    int n_loc = SIZE/npes + (rank < rest);



    #ifdef CUBLAS
    // assign devices to processes and init cublas environment
    int ndevices;
    cudaGetDeviceCount(&ndevices);
    cudaCheckError();
    cudaSetDevice(rank%ndevices);
    cudaCheckError();
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaCheckError();

    // declare device pointers
    double* A_d;
    double* b_col_d;
    double* C_d;
    // parameters for cublasDgemm
    double alpha=1., beta=0.;
    #endif

    // compute the workloads for each process
    int* workloads = (int*) calloc(npes,sizeof(int));
    for (int i=0; i<npes; i++){
	workloads[i] = SIZE/npes + (i < rest);
    }
    // assuming both matrices of size SIZExSIZE for now
    double *A = (double*) malloc(n_loc*SIZE*sizeof(double));
    double *B = (double*) malloc(n_loc*SIZE*sizeof(double));
    double *C = (double*) malloc(n_loc*SIZE*sizeof(double));
    #pragma omp parallel for
    for (int i=0;i<n_loc;i++){
	for (int j=0;j<SIZE;j++){
	    A[i*SIZE + j] = 0;
	    B[i*SIZE + j] = 0;
	    C[i*SIZE + j] = 0;
	}
    }

    // initialise to some data
    timeCPU({init_mat(A,SIZE,rank,npes); eye(B,SIZE,rank,npes);},&t_init);


    // print the matrix A before computation
    #ifdef DEBUG 
    print_mat_parallel(A,rank,SIZE,n_loc,npes,stdout);
    print_mat_parallel(B,rank,SIZE,n_loc,npes,stdout);
    #endif

    // set up the counts and displacements for allgatherv
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


    #ifdef CUBLAS
    // initialise memory on device
    cudaMalloc((void**)&A_d,n_loc*SIZE*sizeof(double));
    cudaCheckError();
    cudaMalloc((void**)&b_col_d,SIZE*ncols*sizeof(double));
    cudaCheckError();
    cudaMalloc((void**)&C_d,n_loc*SIZE*sizeof(double));
    cudaCheckError();
    //cudaMemset(C_d,0,n_loc*SIZE*sizeof(double));
    cudaCheckError();

    timeGPU( cudaMemcpy(A_d,A,n_loc*SIZE*sizeof(double),cudaMemcpyHostToDevice), &t_copy);
    cudaCheckError();
    #endif

    // start the computation
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

	    #ifdef CUBLAS
	    cudaFree((void*)b_col_d);
	    cudaCheckError();
	    cudaMalloc((void**)&b_col_d,SIZE*ncols*sizeof(double));
	    cudaCheckError();
	    #endif
	}


	// create block to send and gather
	//timeCPU( { 
	//	 create_block(B,SIZE,col_idx,n_loc,ncols,b_block);
	//	 MPI_Allgatherv(b_block,n_loc*ncols,MPI_DOUBLE,b_col,rcvcounts,displs,MPI_DOUBLE,MPI_COMM_WORLD); 
	//	}, 
	//	&t_comm);
	timeCPU( create_block(B,SIZE,col_idx,n_loc,ncols,b_block), &t_block);
	timeCPU( MPI_Allgatherv(b_block,n_loc*ncols,MPI_DOUBLE,b_col,rcvcounts,displs,MPI_DOUBLE,MPI_COMM_WORLD), &t_gather );

	#ifdef CUBLAS
	timeGPU( cudaMemcpy(b_col_d,b_col,SIZE*ncols*sizeof(double),cudaMemcpyHostToDevice), &t_copy);
	cudaCheckError();
	#endif

	#if defined(BLAS)
	timeCPU( cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n_loc,ncols,SIZE,1.,A,SIZE,b_col,ncols,0.,C+col_idx,SIZE), &t_compute);
	#elif defined(CUBLAS)
	timeGPU( cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, ncols,n_loc,SIZE,&alpha,b_col_d,ncols,A_d,SIZE,&beta,C_d+col_idx, SIZE), &t_compute);
	cudaCheckError();
	#else
	timeCPU( matmul(A,b_col,n_loc,SIZE,ncols,C,SIZE,col_idx) , &t_compute); 
	#endif

	col_idx += ncols;

    }
    free(b_block);
    free(b_col);

    t_comm = t_block + t_gather;
    printf("proc %d: block = %f, gather = %f\n",rank,t_block,t_comm);

    #ifdef CUBLAS
    timeGPU( cudaMemcpy(C,C_d,n_loc*SIZE*sizeof(double),cudaMemcpyDeviceToHost), &t_copy);
    cudaCheckError();
    cudaFree((void*)A_d);
    cudaCheckError();
    cudaFree((void*)b_col_d);
    cudaCheckError();
    cudaFree((void*)C_d);
    cudaCheckError();

    cublasDestroy_v2(handle);
    cudaCheckError();
    #endif
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
    double* times_copy = NULL;
    if (!rank){
	times_init = (double*) malloc(npes*sizeof(double));
	times_comm = (double*) malloc(npes*sizeof(double));
	times_compute = (double*) malloc(npes*sizeof(double));
	times_copy = (double*) malloc(npes*sizeof(double));
    }
    MPI_Gather(&t_init, 1, MPI_DOUBLE, times_init, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&t_comm, 1, MPI_DOUBLE, times_comm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&t_compute, 1, MPI_DOUBLE, times_compute, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&t_copy, 1, MPI_DOUBLE, times_copy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (!rank){
	double init_mean = 0, comm_mean = 0, compute_mean = 0, copy_mean = 0;
	for (int i = 0;i<npes;i++){
	    init_mean += times_init[i];
	    comm_mean += times_comm[i];
	    compute_mean += times_compute[i];
	    copy_mean += times_copy[i];
	}
	init_mean /= npes;
	comm_mean /= npes;
	compute_mean /= npes;
	copy_mean /= npes;

	double init_std = 0, comm_std = 0, compute_std = 0, copy_std = 0;
	for (int i = 0;i<npes;i++){
	    init_std += (times_init[i] - init_mean) * (times_init[i] - init_mean);
	    comm_std += (times_comm[i] - comm_mean) * (times_comm[i] - comm_mean);
	    compute_std += (times_compute[i] - compute_mean) * (times_compute[i] - compute_mean);
	    copy_std += (times_copy[i] - copy_mean) * (times_copy[i] - copy_mean);
	}
	init_std /= (npes-1);
	comm_std /= (npes-1);
	compute_std /= (npes-1);
	copy_std /= (npes-1);

	init_std = sqrt(init_std);
	comm_std = sqrt(comm_std);
	compute_std = sqrt(compute_std);
	copy_std = sqrt(copy_std);

	//printf("Times:\n\tinit: %.3g s pm %3.g\n\tcomm: %.3g s pm %3.g\n\tcomputation: %.3g s pm %3.g\n\tcopy: %.3g s pm %.3g\n",init_mean,init_std,comm_mean,comm_std,compute_mean,compute_std,copy_mean,copy_std);
	char fname[256];
	char *name = argv[0];
	size_t len = strlen(name);
	name[len-2] = '\0';
	snprintf(fname,sizeof(fname),"%s_%d_times.csv",name,SIZE);
	FILE* outfile = fopen(fname,"a");
	fprintf(outfile, "%d, %f, %f, %f, %f, %f, %f, %f, %f\n", npes,init_mean,init_std,comm_mean,comm_std,compute_mean,compute_std,copy_mean,copy_std);
	fclose(outfile);
    }


    //printf("proc %d:\n\tinit: %.3g s\n\tcomm: %.3g s\n\tcomputation: %.3g s\n",rank,t_init,t_comm,t_compute);

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}
