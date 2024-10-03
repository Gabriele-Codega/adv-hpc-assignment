#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _OPENACC
#include <openacc.h>
#endif

void evolve( double * matrix, double *matrix_new, size_t nloc, size_t dimension ){
  
    //This will be a row dominant program.
    #ifdef _OPENACC
    #pragma acc parallel loop present(matrix,matrix_new) collapse(2) independent
    #else
    #pragma omp parallel for //collapse(2)
    #endif
    for( size_t i = 1 ; i <= nloc; ++i )
        for( size_t j = 1; j <= dimension; ++j )
            matrix_new[ ( i * ( dimension + 2 ) ) + j ] = ( 0.25 ) * 
              ( matrix[ ( ( i - 1 ) * ( dimension + 2 ) ) + j ] + 
        	matrix[ ( i * ( dimension + 2 ) ) + ( j + 1 ) ] + 	  
        	matrix[ ( ( i + 1 ) * ( dimension + 2 ) ) + j ] + 
        	matrix[ ( i * ( dimension + 2 ) ) + ( j - 1 ) ] ); 
}

void save_gnuplot( double *M, size_t dimension){
  
    size_t i , j;
    const double h = 0.1;
    FILE *file;

    file = fopen( "solution.bin", "wb" );

    for( i = 0; i < dimension + 2; ++i ){
	for( j = 0; j < dimension + 2; ++j ){
	    double a = h*j;
	    double b = -h*i;
	    fwrite(&a,sizeof(double),1,file);
	    fwrite(&b,sizeof(double),1,file);
	    fwrite(&M[i*(dimension+2)+j],sizeof(double),1,file);
	}
    }
    fclose( file );
}
void save_gnuplot_distributed(double* A, int rank, int size, int n_loc, int npes){
    if (rank == 0){
	FILE* fh;
	fh = fopen("solution.bin", "wb");
	//fwrite(A,sizeof(double),(n_loc+1)*(size+2),fh);
	double h = 0.1;
	for (int i=0;i<n_loc+1;i++){
	    for (int j=0;j<size+2;j++){
		//fprintf(fh,"%f\t%f\t%f\n",h*j,-h*i,A[i*(size+2)+j]);
		double a = h*j;
		double b = -h*i;
		fwrite(&a,sizeof(double),1,fh);
		fwrite(&b,sizeof(double),1,fh);
		fwrite(&A[i*(size+2)+j],sizeof(double),1,fh);
	    }
	}
	double * buf = (double*) malloc((n_loc+2)*(size+2)*sizeof(double));
	int start = n_loc;
	MPI_Status status;
	int msg_size = 0;
	int nrows = 0;
	for (int count=1;count<npes-1;count++){
            MPI_Recv(buf,(size+2)*(n_loc+2),MPI_DOUBLE,count,100,MPI_COMM_WORLD,&status);
            MPI_Get_count(&status,MPI_DOUBLE,&msg_size);
	    nrows = msg_size/(size+2)-2;
	    //fwrite(buf+size+2,sizeof(double),nrows*(size+2),fh);
	    for (int i=1;i<nrows+1;i++){
		for (int j=0;j<size+2;j++){
		    //fprintf(fh,"%f\t%f\t%f\n",h*j,-h*(i+start),buf[i*(size+2)+j]);
		    double a = h*j;
		    double b = -h*(i+start);
		    fwrite(&a,sizeof(double),1,fh);
		    fwrite(&b,sizeof(double),1,fh);
		    fwrite(&buf[i*(size+2)+j],sizeof(double),1,fh);
		}
	    }
	    start += nrows;
	}
	MPI_Recv(buf,(size+2)*(n_loc+2),MPI_DOUBLE,npes-1,100,MPI_COMM_WORLD,&status);
	MPI_Get_count(&status,MPI_DOUBLE,&msg_size);
	nrows = msg_size/(size+2)-2;
	//fwrite(buf+size+2,sizeof(double),(n_loc+1)*(size+2),fh);
	for (int i=1;i<nrows+2;i++){
	    for (int j=0;j<size+2;j++){
		//fprintf(fh,"%f\t%f\t%f\n",h*j,-h*(i+start),buf[i*(size+2)+j]);
		double a = h*j;
		double b = -h*(i+start);
		fwrite(&a,sizeof(double),1,fh);
		fwrite(&b,sizeof(double),1,fh);
		fwrite(&buf[i*(size+2)+j],sizeof(double),1,fh);
	    }
	}
	fclose(fh);
	free(buf);
    } else MPI_Send(A,(size+2)*(n_loc+2),MPI_DOUBLE,0,100,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
}

void save_gnuplot_mpi (double*matrix, int rank, int dimension, int n_loc, int npes, int rest){
      MPI_File fh;
      MPI_File_open(MPI_COMM_WORLD, "solution.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);
      if (rank==0){
          MPI_Offset start = 0;
          MPI_File_write_at_all(fh,start,matrix,(n_loc+1)*(dimension+2),MPI_DOUBLE,MPI_STATUS_IGNORE);
      }else if (rank == npes-1){
          MPI_Offset start = n_loc*(dimension+2)*rank; //base work for each proc
          start += (dimension+2); //process 0 always writes one extra line
          if (rank>= rest){
              start += rest;
          } //extra work for processes in the 

          MPI_File_write_at_all(fh,start*sizeof(double),&matrix[dimension+2],(n_loc+1)*(dimension+2),MPI_DOUBLE,MPI_STATUS_IGNORE);
      }else{
          MPI_Offset start = n_loc*(dimension+2)*rank; //base work for each proc
          start += (dimension+2); //process 0 always writes one extra line
          if (rank>= rest){
              start += rest;
          } //extra work for processes in the 
          MPI_File_write_at_all(fh,start*sizeof(double),&matrix[dimension+2],n_loc*(dimension+2),MPI_DOUBLE,MPI_STATUS_IGNORE);
      }
      MPI_File_close(&fh);
}

// A Simple timer for measuring the walltime
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
