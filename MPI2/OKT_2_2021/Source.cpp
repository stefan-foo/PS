#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <time.h>
#include <iomanip>

#define N 8

int main(int argc, char** argv)
{
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int A[N][N];
	int B[N][N];
	int C[N][N];

	int q = sqrt(size);
	int s = N / q;

	//if (q * q != size) return -1;

	if (rank == 0) {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				A[i][j] = i * N + j;
				B[i][j] = j;
			}
		}

		std::cout << "MATRIX A" << "\n";
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				std::cout << std::setw(3) << A[i][j] << " ";
			}
			std::cout << "\n";
		}

		std::cout << "MATRIX B" << "\n";
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				std::cout << std::setw(3) << B[i][j] << " ";
			}
			std::cout << "\n";
		}

		std::cout << "MATRIX C" << "\n";
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j)
			{
				C[i][j] = 0;
				for (int k = 0; k < N; ++k)
				{
					C[i][j] += A[i][k] * B[k][j];
				}
				std::cout << std::setw(3) << C[i][j] << " ";
				C[i][j] = 0;
			}
			std::cout << "\n";
		}

		std::cout << std::endl;
	}

	//MPI_Barrier(MPI_COMM_WORLD);

	int col_rank, row_rank;

	MPI_Comm row_comm, col_comm;
	MPI_Comm_split(MPI_COMM_WORLD, rank / q, 1, &row_comm);
	MPI_Comm_split(MPI_COMM_WORLD, rank % q, 1, &col_comm);

	MPI_Comm_rank(row_comm, &row_rank);
	MPI_Comm_rank(col_comm, &col_rank);

	MPI_Datatype temp, rows_split, col;
	MPI_Type_vector(s, N, N * q, MPI_INT, &temp);
	MPI_Type_create_resized(temp, 0, N * sizeof(int), &rows_split);
	
	MPI_Type_vector(s, 1, q, MPI_INT, &temp);
	MPI_Type_create_resized(temp, 0, sizeof(int) * N, &col);
	MPI_Type_vector(N, 1, 1, col, &temp);
	MPI_Type_create_resized(temp, 0, sizeof(int), &col);
	
	MPI_Type_commit(&rows_split);
	MPI_Type_commit(&col);

	int* loc_a = new int[s * N];
	int* loc_b = new int[s * N];
	int* loc_c = new int[s * s];

	if (row_rank == 0) {
		MPI_Scatter(A, 1, rows_split, loc_a, s * N, MPI_INT, 0, col_comm);
	}
	if (col_rank == 0) {
		MPI_Scatter(B, 1, col, loc_b, s * N, MPI_INT, 0, row_comm);
	}

	MPI_Bcast(loc_a, s * N, MPI_INT, 0, row_comm);
	MPI_Bcast(loc_b, s * N, MPI_INT, 0, col_comm);

	//std::cout << "PROCESS " << rank << " " << row_rank << " " << col_rank << "\n";
	//for (int i = 0; i < s; i++) {
	//	for (int j = 0; j < N; j++) {
	//		std::cout << loc_a[i * N + j] << " ";
	//	}
	//	std::cout << "\n";
	//}
	//for (int i = 0; i < N; i++) {
	//	for (int j = 0; j < s; j++) {
	//		std::cout << std::setw(3) << loc_b[i * s + j] << " ";
	//	}
	//	std::cout << "\n";
	//}

	for (int i = 0; i < s; i++) {
		for (int j = 0; j < s; j++) {
			loc_c[i * s + j] = 0;
			for (int k = 0; k < N; k++) {
				loc_c[i * s + j] += loc_a[i * N + k] * loc_b[k * s + j];
			}
		}
	}
	
	MPI_Type_vector(s, s, s * q, MPI_INT, &temp);
	MPI_Type_create_resized(temp, 0, s * sizeof(int), &rows_split);
	MPI_Type_commit(&rows_split);

	int* part_c = new int[s * N];

	MPI_Gather(loc_c, s * s, MPI_INT, part_c, 1, rows_split, 0, col_comm);

	if (col_rank == 0) {
		MPI_Gather(part_c, s * N, MPI_INT, C, 1, col, 0, row_comm);

		if (row_rank == 0) {
			std::cout << "MATRIX C" << "\n";
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < N; j++) {
					std::cout << std::setw(3) << C[i][j] << " ";
				}
				std::cout << "\n";
			}
		}
	}

	MPI_Finalize();
	return 0;
}