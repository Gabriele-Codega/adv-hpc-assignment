#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>

#include "utils.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _OPENACC
#include <openacc.h>
#endif


int main(int argc, char* argv[]){

    #ifdef _OPENMP
    int provided;
    MPI_Init_thread(NULL,NULL,MPI_THREAD_FUNNELED,&provided);
    if (provided != MPI_THREAD_FUNNELED){
	printf("requested th level %d but got %d instead\n",MPI_THREAD_FUNNELED,provided);
    }
    #pragma omp parallel
    {
	int nth = omp_get_num_threads();
    #pragma omp master
	printf("using OMP with %d threads\n",nth);
    }
    #else
    MPI_Init(NULL,NULL);
    #endif

    int rank, npes;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&npes);

    #ifdef _OPENACC
    int ngpus = acc_get_num_devices(acc_device_nvidia);
    acc_set_device_num(rank%ngpus,acc_device_nvidia);
    acc_init(acc_device_nvidia);
    #endif

    // initialize matrix
    double *matrix, *matrix_new;
    #ifndef _OPENACC
    double *tmp_matrix;
    #endif
    size_t dimension = 0, iterations = 0, row_peek = 0, col_peek = 0;
    size_t byte_dimension = 0;

    // check on input parameters
    if(argc != 5) {
	fprintf(stderr,"\nwrong number of arguments. Usage: ./a.out dim it n m\n");
	return 1;
    }

    dimension = atoi(argv[1]);
    iterations = atoi(argv[2]);
    row_peek = atoi(argv[3]);
    col_peek = atoi(argv[4]);

    int rest = dimension%npes;
    int n_loc = dimension/npes + (rank<rest);
    int offset = rest * (rank >= rest);

    if (!rank){
	printf("matrix size = %zu\n", dimension);
	printf("number of iterations = %zu\n", iterations);
	printf("element for checking = Mat[%zu,%zu]\n",row_peek, col_peek);
    }

    if((row_peek > n_loc) || (col_peek > dimension)){
	fprintf(stderr, "Cannot Peek a matrix element outside of the matrix dimension\n");
	fprintf(stderr, "Arguments n and m must be smaller than %zu,%zu\n", n_loc, dimension);
	return 1;
    }


    double t_init = 0, t_comm = 0, t_comp = 0, t_io = 0;
    // allocate matrix chunks 
    byte_dimension = sizeof(double) * ( n_loc + 2 ) * ( dimension + 2 );
    matrix = ( double* )malloc( byte_dimension );
    matrix_new = ( double* )malloc( byte_dimension );

    double t1 = MPI_Wtime(); //start time init
    memset( matrix, 0, byte_dimension );
    memset( matrix_new, 0, byte_dimension );

    //fill initial values  
    #pragma omp parallel for collapse(2)
    for(int i = 1; i <= n_loc; ++i )
	for(int j = 1; j <= dimension; ++j )
	    matrix[ ( i * ( dimension + 2 ) ) + j ] = 0.5;

    // set up borders 
    double increment = 100.0 / ( dimension + 1 );

    // increment is the same for everyone but the offset changes:
    // global_row = offset + i
    // Should iterate over rows in the inside of the domain

    // start is the global index of the first row
    int start = n_loc*rank+offset;
    if (rank == npes-1){
	#pragma omp parallel for
	for(int i=1; i <= n_loc+1; ++i ){
	    // first col -- border, not inside
	    matrix[ i * ( dimension + 2 ) ] = (start+i) * increment;
	    matrix_new[ i * ( dimension + 2 ) ] = (start+i)* increment;
	}

	#pragma omp parallel for
	for(int i=1; i <= dimension+1; ++i ){
	    // last row -- border, not inside
	    matrix[ ( ( n_loc + 1 ) * ( dimension + 2 ) ) + ( dimension + 1 - i ) ] = i * increment;
	    matrix_new[ ( ( n_loc + 1 ) * ( dimension + 2 ) ) + ( dimension + 1 - i ) ] = i * increment;
	}
    }else{
	#pragma omp parallel for
	for(int i=1; i < n_loc+1; ++i ){
	    // first col -- border, not inside
	    matrix[ i * ( dimension + 2 ) ] = (start+i) * increment;
	    matrix_new[ i * ( dimension + 2 ) ] = (start+i)* increment;
	}

    }

    double t2 = MPI_Wtime(); // end time init
    t_init = t2-t1;

    //print_mat_parallel(matrix, rank, dimension+2, n_loc+2,npes,stdout);


    int prev = (rank == 0) ? MPI_PROC_NULL : rank-1;
    int next = (rank == npes-1) ? MPI_PROC_NULL : rank+1;

    int tag_prev = (rank == 0) ? -1 : rank-1;
    int tag_next = (rank == npes-1) ? -1 : rank+1;


    int total_size = (n_loc+2)*(dimension+2);
    #pragma acc data copy(matrix[0:total_size]) copyin(matrix_new[0:total_size])
    {
    // start algorithm
    for(int it = 0; it < iterations; ++it ){
	
	// send to previous and receive from next, then vice versa
	double t3 = MPI_Wtime(); // start time communication
	#pragma acc host_data use_device(matrix)
	{
	    MPI_Sendrecv(&matrix[dimension+2], dimension+2, MPI_DOUBLE, prev, rank,\
		  &matrix[(n_loc+1)*(dimension+2)], dimension+2, MPI_DOUBLE, next, tag_next,\
		  MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    MPI_Sendrecv(&matrix[(n_loc)*(dimension+2)], dimension+2, MPI_DOUBLE, next, rank,\
		  matrix, dimension+2, MPI_DOUBLE, prev, tag_prev,\
		  MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	double t4 = MPI_Wtime(); //end time communication
	t_comm += (t4-t3);
      
	double t5 = MPI_Wtime(); //start time computation (includes swapping of matrices or ponters)
	evolve( matrix, matrix_new,n_loc, dimension );

	#ifdef _OPENACC
	#pragma acc parallel loop present(matrix[0:total_size],matrix_new[0:total_size]) collapse(2) independent
	for (int i=1;i<=n_loc;i++){
	    for (int j=1;j<=dimension;j++){
		matrix[i*(dimension+2)+j] = matrix_new[i*(dimension+2)+j];
	    }
	}
	#else
	// swap the pointers
	tmp_matrix = matrix;
	matrix = matrix_new;
	matrix_new = tmp_matrix;
	#endif
	double t6 = MPI_Wtime(); //end time computation
	t_comp += (t6-t5);

    }
    }
    
    //print_mat_parallel(matrix, rank, dimension+2, n_loc+2,npes,stdout);
    
    if (!rank){
	printf( "\nmatrix[%zu,%zu] = %f\n", row_peek, col_peek, matrix[ ( row_peek + 1 ) * ( dimension + 2 ) + ( col_peek + 1 ) ] );
    }

    double t7 = MPI_Wtime();
    if (npes>1){ 
	//save_gnuplot_distributed( matrix, rank, dimension, n_loc, npes );
	save_gnuplot_mpi(matrix,rank,dimension,n_loc,npes,rest);
    }else{
	save_gnuplot( matrix, dimension);
    }
    double t8 = MPI_Wtime();
    t_io = (t8-t7);

    free( matrix );
    free( matrix_new );

    // gather times and compute statistics
    double* times_init = NULL;
    double* times_comm = NULL;
    double* times_comp = NULL;
    double* times_io = NULL;
    if (!rank){
	times_init = (double*) malloc(npes*sizeof(double));
	times_comm = (double*) malloc(npes*sizeof(double));
	times_comp = (double*) malloc(npes*sizeof(double));
	times_io = (double*) malloc(npes*sizeof(double));
    }
    MPI_Gather(&t_init, 1, MPI_DOUBLE, times_init, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&t_comm, 1, MPI_DOUBLE, times_comm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&t_comp, 1, MPI_DOUBLE, times_comp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&t_io, 1, MPI_DOUBLE, times_io, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (!rank){
	double init_mean = 0, comm_mean = 0, compute_mean = 0, io_mean = 0;
	for (int i = 0;i<npes;i++){
	    init_mean += times_init[i];
	    comm_mean += times_comm[i];
	    compute_mean += times_comp[i];
	    io_mean += times_io[i];
	}
	init_mean /= npes;
	comm_mean /= npes;
	compute_mean /= npes;
	io_mean /= npes;

	double init_std = 0, comm_std = 0, compute_std = 0, io_std = 0;
	for (int i = 0;i<npes;i++){
	    init_std += (times_init[i] - init_mean) * (times_init[i] - init_mean);
	    comm_std += (times_comm[i] - comm_mean) * (times_comm[i] - comm_mean);
	    compute_std += (times_comp[i] - compute_mean) * (times_comp[i] - compute_mean);
	    io_std += (times_io[i] - io_mean) * (times_io[i] - io_mean);
	}
	init_std /= (npes-1);
	comm_std /= (npes-1);
	compute_std /= (npes-1);
	io_std /= (npes-1);

	init_std = sqrt(init_std);
	comm_std = sqrt(comm_std);
	compute_std = sqrt(compute_std);
	io_std = sqrt(io_std);

	char fname[256];
	char *name = argv[0];
	size_t len = strlen(name);
	name[len-2] = '\0';
	snprintf(fname,sizeof(fname),"%s_%d_times.csv",name,dimension);
	FILE* outfile = fopen(fname,"a");
	fprintf(outfile, "%d, %f, %f, %f, %f, %f, %f, %f, %f\n", npes,init_mean,init_std,comm_mean,comm_std,compute_mean,compute_std,io_mean,io_std);
	fclose(outfile);

	free(times_init);
	free(times_comm);
	free(times_comp);
	free(times_io);

    }

    MPI_Finalize();

    return 0;
}

