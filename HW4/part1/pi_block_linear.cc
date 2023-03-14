#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>
#define SEED 326172

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: init MPI
    int i;
    int count = 0;
    double x, y, z;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);    
    long long int times = tosses / world_size;
    int recieved[world_size];
    int tmp;

    unsigned int seed = SEED * world_rank;
    for (i = 0; i < times; i++)           
    {
        x = ((double)rand_r(&seed)) / RAND_MAX;       
        y =((double)rand_r(&seed)) / RAND_MAX;          
        z = x*x + y*y;          
        if (z <= 1)
        {
            count++;             
        }
    }

    if (world_rank > 0)
    {
        MPI_Send(&count, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: master
        recieved[0] = count;
        for(i = 1; i < world_size; i++)
        {
            MPI_Recv(&tmp, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
            recieved[i] = tmp;
        }
    }
    if (world_rank == 0)
    {
        // TODO: process PI result
        int finalcount = 0;
        for(i = 0; i < world_size; i++)
		{
			finalcount += recieved[i];
		}
        pi_result = ((double)finalcount/(double)tosses)*4.0;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}