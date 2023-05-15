#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <time.h>

#define K 8
#define N 6
#define M 10

int main(int argc, char** argv)
{
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int A[K][N];
	int B[N][M];
	int C[K][M];

	int s = K / size;

	if (rank == 0) {
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < N; j++) {
				A[i][j] = i * 10 + j;
			}
		}

		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {
				B[i][j] = j % 2;
			}
		}

		std::cout << "MATRIX A: \n";
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < N; j++) {
				std::cout << A[i][j] << " ";
			}
			std::cout << "\n";
		}

		std::cout << "\nMATRIX B: \n";
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {
				std::cout << B[i][j] << " ";
			}
			std::cout << "\n";
		}
		std::cout << std::endl;
	}

	MPI_Datatype temp, pth_row;
	MPI_Type_vector(s, N, N * size, MPI_INT, &temp);
	MPI_Type_create_resized(temp, 0, sizeof(int) * N, &pth_row);
	MPI_Type_commit(&pth_row);

	int* loc_a = new int[s * N];
	int* loc_c = new int[s * M];
	
	MPI_Bcast(B, N * M, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatter(A, 1, pth_row, loc_a, N * s, MPI_INT, 0, MPI_COMM_WORLD);

	for (int i = 0; i < s; i++) {
		for (int j = 0; j < M; j++) {
			loc_c[i * M + j] = 0;
			for (int k = 0; k < N; k++) {
				loc_c[i * M + j] += loc_a[i * N + k] * B[k][j];
			}
		}
	}

	MPI_Datatype pth_res;
	MPI_Type_vector(s, M, M * size, MPI_INT, &temp);
	MPI_Type_create_resized(temp, 0, sizeof(int) * M, &pth_res);
	MPI_Type_commit(&pth_res);

	MPI_Gather(loc_c, s * M, MPI_INT, C, 1, pth_res, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		std::cout << "RESULT: \n";
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < M; j++) {
				std::cout << C[i][j] << " ";
			}
			std::cout << "\n";
		}
	}

	MPI_Finalize();
	return 0;
}