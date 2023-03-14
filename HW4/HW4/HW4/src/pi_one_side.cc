#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#define SEED 326172

int finalcount = 0;

int fnz (int *received, int *oldreceived, int size)
{

    int diff = 0;
    for (int i = 0; i < size; i++)
       diff |= (received[i] != oldreceived[i]);

    if (diff)
    {
       int res = 0;
       for (int i = 0; i < size; i++)
       {
            if(received[i] >0)
            {
                res ++;
            }
            if(received[i] != oldreceived[i])
            {
                finalcount += received[i];
            }
            oldreceived[i] = received[i];
       }
       return(res == size);
    }
    return 0;
}


int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    // TODO: MPI init
    int i;
    double x, y, z;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); 
    long long int times = tosses / world_size;
    int count = 0;

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

    if (world_rank == 0)
    {
        // Master
        int *oldreceived = (int *)malloc(world_size * sizeof(int));

        int *received;
        
        MPI_Alloc_mem(world_size * sizeof(int), MPI_INFO_NULL, &received);
        
        
        for (int i = 0; i < world_size; i++)
        {
            received[i] = 0;
            oldreceived[i] = 0;
        }
        received[0] = count;
        printf("before %d\n", world_rank);
        MPI_Win_create(received, world_size * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        printf("after %d\n", world_rank);

        int ready = 0;
        while (!ready)
        {
            MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
            ready = fnz(received, oldreceived, world_size);
            MPI_Win_unlock(0, win);
        }

        MPI_Win_free(&win);
        MPI_Free_mem(received);
        free(oldreceived);
    }
    else
    {
        // Workers
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Put(&count, 1, MPI_INT, 0, world_rank, 1, MPI_INT, win);
        MPI_Win_unlock(0, win);

        MPI_Win_free(&win);
    }

    

    if (world_rank == 0)
    {
        // TODO: handle PI result
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