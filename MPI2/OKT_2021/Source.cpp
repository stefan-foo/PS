#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <time.h>

#define N 8

int main(int argc, char** argv)
{
	int rank, p;
	int A[N][N];
	int B[N];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	int q = sqrt(p);
	int s = N / q;

	if (q * q != p) {
		exit(-1);
	}

	if (rank == 0) {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				A[i][j] = i * N + j;
			}
			B[i] = 1;
		}
	}

	int row_ind = rank / q;
	int col_ind = rank % q;

	MPI_Comm col_comm, row_comm;
	MPI_Comm_split(MPI_COMM_WORLD, col_ind, row_ind, &col_comm);
	MPI_Comm_split(MPI_COMM_WORLD, row_ind, col_ind, &row_comm);

	int row_rank, col_rank;
	MPI_Comm_rank(col_comm, &col_rank);
	MPI_Comm_rank(row_comm, &row_rank);

	MPI_Datatype v_split, temp;
	MPI_Type_vector(s, 1, q, MPI_INT, &temp);
	MPI_Type_create_resized(temp, 0, sizeof(int), &v_split);
	MPI_Type_commit(&v_split);

	MPI_Datatype mat_block, block_temp;
	MPI_Type_vector(s, 1, N * q, v_split, &block_temp);
	MPI_Type_create_resized(block_temp, 0, s, &mat_block);
	MPI_Type_commit(&mat_block);

	int* loc_A = new int[s * s];
	int* loc_B = new int[s];

	if (col_rank == 0) {
		MPI_Scatter(B, 1, v_split, loc_B, s, MPI_INT, 0, row_comm);
	}

	MPI_Bcast(loc_B, s, MPI_INT, 0, col_comm);

	//std::cout << "RANK - " << rank << " " << col_rank << ": ";
	//for (int i = 0; i < s; i++) {
	//	std::cout << loc_B[i] << " ";
	//}
	//std::cout << std::endl;

	if (rank == 0) {
		for (int i = 0; i < s; i++) {
			for (int j = 0; j < s; j++) {
				loc_A[i * s + j] = A[i * q][j * q];
			}
		}

		for (int i = 1; i < p; i++) {
			MPI_Send(&A[(i / q)][i % q], 1, mat_block, i, 0, MPI_COMM_WORLD);
		}
	}
	else {
		MPI_Recv(loc_A, s * s, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	std::cout << "RANK - " << rank << ": \n";
	for (int i = 0; i < s; i++) {
		for (int j = 0; j < s; j++) {
			std::cout << loc_A[i * s + j] << " ";
		}
		std::cout << "\n";
	}
	std::cout << std::endl;

	int* loc_res = new int[s];
	int* sum_res = new int[s];
	int* c = new int[N];
	for (int i = 0; i < s; i++) {
		loc_res[i] = 0;
		for (int j = 0; j < s; j++) {
			loc_res[i] += loc_A[i * s + j] * loc_B[j];
		}
	}

	MPI_Reduce(loc_res, sum_res, s, MPI_INT, MPI_SUM, 0, row_comm);

	if (row_rank == 0) {
		MPI_Gather(sum_res, s, MPI_INT, c, 1, v_split, 0, col_comm);

		if (col_rank == 0) {
			std::cout << "RESULT:\n";
			for (int i = 0; i < N; i++) {
				std::cout << c[i] << " ";
			}
		}
	}


	MPI_Finalize();
	return 0;
}