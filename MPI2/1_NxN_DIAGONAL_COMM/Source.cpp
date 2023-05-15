#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <time.h>

constexpr int MASTER_DIAG = 0;

int main(int argc, char** argv)
{
	int world_rank, size, diagonal_rank;
	MPI_Group group_world, group_diagonal;
	MPI_Comm diagonal_comm;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_group(MPI_COMM_WORLD, &group_world);

	int N = sqrt(size);

	int* incl = new int[N];

	for (int i = 0; i < N; i++) {
		incl[i] = i * (N + 1);
	}

	MPI_Group_incl(group_world, N, incl, &group_diagonal);
	MPI_Comm_create(MPI_COMM_WORLD, group_diagonal, &diagonal_comm);
	MPI_Group_rank(group_diagonal, &diagonal_rank);

	delete[] incl;

	if (diagonal_rank >= 0 /*world_rank % (N + 1) == 0*/) {
		int buffer;

		if (diagonal_rank == MASTER_DIAG) {
			buffer = 15;
		}

		MPI_Bcast(&buffer, 1, MPI_INT, MASTER_DIAG, diagonal_comm);
		std::cout << "Primljena poruka " << buffer << " WORLD_RANK[" << world_rank << "] DIAG_RANK[" << diagonal_rank << "]" << std::endl;
	}

	MPI_Finalize();
	return 0;
}