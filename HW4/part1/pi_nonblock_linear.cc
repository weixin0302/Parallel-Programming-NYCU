#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
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

    // TODO: MPI init
    int i;
    int count = 0;
    double x, y, z;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);    
    long long int times = tosses / world_size;
    int recieved[world_size];
    int tmp;
    MPI_Request requests[world_size-1];
    MPI_Status status[world_size-1];

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
        // TODO: MPI workers
        MPI_Send(&count, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        recieved[0] = count;
        for(i = 1; i < world_size; i++)
        {
            MPI_Irecv(&recieved[i], 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &requests[i-1]);
        }
        MPI_Waitall(world_size-1,requests,status);
    }

    if (world_rank == 0)
    {
        // TODO: PI result
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
