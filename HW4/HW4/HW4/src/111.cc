#include <stdio.h>
#include <mpi.h>
int main(int argc, char **argv)
{
    long i, n=0, rects_par_proc=0, my_deb=0, my_end=0;
    double x, h, sum=0.0, pi=0.0;
    int myrank, nbprocs;
    MPI_Win win_pi, win_n;
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &myrank );
    MPI_Comm_size( MPI_COMM_WORLD, &nbprocs);
    /* Lecture of n from keyboard */
    if (myrank==0) {
        MPI_Win_create(&n, sizeof(long), sizeof(long),
        MPI_INFO_NULL, MPI_COMM_WORLD, &win_n);
        MPI_Win_create(&pi, sizeof(double), sizeof(double),
        MPI_INFO_NULL, MPI_COMM_WORLD, &win_pi);
    }
    else {
        MPI_Win_create(MPI_BOTTOM, 0, sizeof(long),
        MPI_INFO_NULL, MPI_COMM_WORLD, &win_n);
        MPI_Win_create(MPI_BOTTOM, 0, sizeof(double),
        MPI_INFO_NULL, MPI_COMM_WORLD, &win_pi);
    }
    MPI_Win_fence(0, win_n);
    if (myrank != 0)
        MPI_Get(&n, 1, MPI_LONG, 0, 0, 1, MPI_LONG, win_n);
    MPI_Win_fence(0, win_n);

    h = 1.0 / n; 
    rects_par_proc = n / nbprocs;
    my_deb = myrank*rects_par_proc; 
    my_end = my_deb+rects_par_proc;
    for (i=my_deb; i<my_end; i++) {
        x = (i+0.5)*h; 
        sum += 4.0 / (1.0 + x*x);
    }
    pi = h * sum ;
    MPI_Win_fence(0, win_pi);
    if (myrank)
        MPI_Accumulate(&pi, 1, MPI_DOUBLE, 0, 0, 1, MPI_DOUBLE, MPI_SUM, win_pi);
    MPI_Win_fence(0, win_pi);
    if (myrank==0) printf("Piis approximatly %0.16f\n", pi);
    MPI_Finalize(); 
    return 0; 
}