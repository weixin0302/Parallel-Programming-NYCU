# include <stdio.h>
# include <stdlib.h>
# include <mpi.h>

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr)
{
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if(world_rank == 0)
    {
        scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
	    int n = *n_ptr;
        int m = *m_ptr;
        int l = *l_ptr;

        *a_mat_ptr = (int*)malloc(sizeof(int) * n * m);
	    *b_mat_ptr = (int*)malloc(sizeof(int) * m * l);
        
        for (int i = 0; i < n; i++){
	        for (int j = 0; j < m; j++){
                int *tmp = *a_mat_ptr + i * m + j;
		        scanf("%d", tmp);
            }
	    }
        for (int i = 0; i < m; i++){
	        for (int j = 0; j < l; j++){
		        int *tmp = *b_mat_ptr + i * l + j;
		        scanf("%d", tmp);
            }
        }
    }
}

void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat)
{
    int world_rank, world_size;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0)
    {
        int *c_mat;
    	c_mat = (int*)malloc(sizeof(int) * n * l);

        int averow = n / (world_size-1);
        int extra = n % (world_size-1);
        int offset = 0;
        int rows;

        for (int worker = 1; worker <= world_size-1; worker++){
            MPI_Send(&n, 1, MPI_INT, worker, 1, MPI_COMM_WORLD);
            MPI_Send(&m, 1, MPI_INT, worker, 1, MPI_COMM_WORLD);
            MPI_Send(&l, 1, MPI_INT, worker, 1, MPI_COMM_WORLD);
            rows = (worker <= extra)? averow + 1: averow;
            MPI_Send(&offset, 1, MPI_INT, worker, 1, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, worker, 1, MPI_COMM_WORLD);
            MPI_Send(&a_mat[offset * m], rows * m, MPI_INT, worker, 1, MPI_COMM_WORLD);
            MPI_Send(&b_mat[0], m * l, MPI_INT, worker, 1, MPI_COMM_WORLD);
            offset += rows;
        }

        for(int worker = 1; worker <= world_size-1; worker++)
        {
            int source = worker;
            MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&c_mat[offset * l], rows * l, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
        }

        for(int i=0; i<n; i++)
        {
            for(int j=0; j<l; j++)
            {
                printf("%d ", c_mat[i * l + j]);
            }
            printf("\n");
        }
        free(c_mat);
    }

    else if(world_rank > 0)
    {
        int N, M, L;
        int *a, *b, *c;
        int offset, rows;

        MPI_Recv(&N, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&M, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&L, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

    	a = (int*)malloc(sizeof(int) * N * M);
    	b = (int*)malloc(sizeof(int) * M * L);
    	c = (int*)malloc(sizeof(int) * N * L);

        MPI_Recv(&offset, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&a[0], rows * M, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&b[0], M * L, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

        for (int k = 0; k < L; k++){
            for (int i = 0; i < rows; i++){
                c[i * L + k] = 0;
                for (int j = 0; j < M; j++){
                    c[i * L + k] += a[i * M + j] * b[j * L + k];
                }
            }
        }

        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&c[0], rows * L, MPI_INT, 0, 2, MPI_COMM_WORLD);

        free(a);
        free(b);
        free(c);
    }
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0)
    {
        free(a_mat);
        free(b_mat);
    }
}