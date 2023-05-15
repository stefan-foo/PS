#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <time.h>
#include <iomanip>

constexpr int MASTER = 0;
constexpr int N = 8;

int main(int argc, char** argv)
{
	int rank, size;
	int A[N][N], B[N], *local_A, *local_B;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int q = (int)sqrt(size);
	if (q * q != size) {
		exit(1);
	}
	int k = N / q;

	local_A = new int[k * k];
	local_B = new int[k];

	if (rank == MASTER) {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				A[i][j] = i + j;
			}
			B[i] = i;
		}

		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				std::cout << std::setw(3) << A[i][j] << " ";
			}
			std::cout << std::endl;
		}
	}

	MPI_Datatype matBlock;
	MPI_Type_vector(k, k, N, MPI_INT, &matBlock);
	MPI_Type_commit(&matBlock);

	if (rank == MASTER) {
		int x = MASTER / q;
		int y = MASTER % q;

		for (int i = 0; i < q; i++) {
			for (int j = 0; j < q; j++) {
				if (i != x || j != y) {
					MPI_Send(&A[i * k][j * k], 1, matBlock, i * q + j, 0, MPI_COMM_WORLD);
				}
			}
		}

		for (int i = 0; i < k; i++) {
			for (int j = 0; j < k; j++) {
				local_A[i * k + j] = A[x * k + i][y * k + j];
			}
		}
	}
	else {
		MPI_Recv(local_A, k * k, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	MPI_Comm col_comm, row_comm;
	MPI_Comm_split(MPI_COMM_WORLD, rank / q, rank % q, &row_comm);
	MPI_Comm_split(MPI_COMM_WORLD, rank % q, rank / q, &col_comm);

	int col_rank, row_rank;
	MPI_Comm_rank(col_comm, &col_rank);
	MPI_Comm_rank(row_comm, &row_rank);

	if (col_rank == 0) {
		MPI_Datatype b_portion, b_portion_resized;
		MPI_Type_vector(k, 1, q, MPI_INT, &b_portion);
		MPI_Type_create_resized(b_portion, 0, sizeof(MPI_INT), &b_portion_resized);
		MPI_Type_commit(&b_portion_resized);

		MPI_Scatter(B, 1, b_portion_resized, local_B, k, MPI_INT, 0, row_comm);
	}

	MPI_Bcast(local_B, k, MPI_INT, 0, col_comm);

	std::cout << "RANK: " << rank << " " << rank / q << " " << rank % q << "\n";
	for (int i = 0; i < k * k; i++) {
		std::cout << local_A[i] << " ";
	}
	std::cout << "\n";
	for (int j = 0; j < k; j++) {
		std::cout << local_B[j] << " ";
	}
	std::cout << std::endl;
	
	MPI_Finalize();
	return 0;
}