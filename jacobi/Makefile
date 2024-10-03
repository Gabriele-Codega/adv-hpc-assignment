
all: base cpu gpu rma

base:
	gcc -O3 jacobi_base.c -o ${@}.x
cpu: utils.c
	mpiicc -O3 -qopenmp jacobi_parallel.c $^ -o ${@}.x 
gpu: utils.c
	mpicc -O3 -mp -acc jacobi_parallel.c $^ -o ${@}.x 
rma: utils.c
	mpicc -O3 -mp -acc jacobi_rma.c $^ -o ${@}.x 

clean:
	-rm -f *.o
	-rm -f *.x
