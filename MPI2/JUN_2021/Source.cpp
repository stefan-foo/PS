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

	int q = sqrt(size);
	if (q * q != size) return -1;
	int s = N / q;

	int A[N][N];
	int B[N];
	int C[N];

	if (rank == 0) {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				A[i][j] = i * 10 + j;
			}
			B[i] = i;
		}
		
		std::cout << "MATRIX A:\n";
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				std::cout << std::setw(3) << A[i][j] << " ";
			}
			std::cout << "\n";
		}

		std::cout << "VECTOR B:\n";
		for (int i = 0; i < N; i++) {
			std::cout << B[i] << " ";
		}
		std::cout << std::endl;
	}

	int* loc_a = new int[s * s];
	int* loc_b = new int[s];
	
	MPI_Datatype temp, block;
	MPI_Type_vector(s, s, q * N, MPI_INT, &block);
	MPI_Type_commit(&block);

	if (rank == 0) {
		for (int i = 0; i < s; i++) {
			for (int j = 0; j < s; j++) {
				loc_a[i * s + j] = A[i][j];
			}
		}

		int p = 0;
		for (int i = 0; i < q; i++) {
			for (int j = 0; j < q; j++) {
				if (p != 0)
					MPI_Send(&A[i][j * s], 1, block, p, 0, MPI_COMM_WORLD);
				p++;
			}
		}
	}
	else {
		MPI_Recv(loc_a, s * s, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	/*std::cout << "RANK " << rank << "\n";
	for (int i = 0; i < s; i++) {
		for (int j = 0; j < s; j++) {
			std::cout << 
		}
	}*/
	int row_rank, col_rank;

	MPI_Comm row_comm, col_comm;
	MPI_Comm_split(MPI_COMM_WORLD, rank / q, rank % q, &row_comm);
	MPI_Comm_split(MPI_COMM_WORLD, rank % q, rank / q, &col_comm);

	MPI_Comm_rank(row_comm, &row_rank);
	MPI_Comm_rank(col_comm, &col_rank);

	if (col_rank == 0) {
		MPI_Scatter(B, s, MPI_INT, loc_b, s, MPI_INT, 0, row_comm);
	}
	MPI_Bcast(loc_b, s, MPI_INT, 0, col_comm);

	int* loc_c = new int[s];

	for (int i = 0; i < s; i++) {
		loc_c[i] = 0;
		for (int j = 0; j < s; j++) {
			loc_c[i] += loc_a[i * s + j] * loc_b[j];
		}
	}

	int* part_res = new int[s];

	MPI_Datatype gather_res;
	MPI_Type_vector(s, 1, q, MPI_INT, &temp);
	MPI_Type_create_resized(temp, 0, sizeof(int), &gather_res);
	MPI_Type_commit(&gather_res);

	MPI_Reduce(loc_c, part_res, s, MPI_INT, MPI_SUM, 0, row_comm);

	if (row_rank == 0) {
		MPI_Gather(part_res, s, MPI_INT, C, 1, gather_res, 0, col_comm);

		if (col_rank == 0) {
			for (int i = 0; i < N; i++) {
				std::cout << C[i] << "\n";
			}
			std::cout << std::endl;
		}
	}

	MPI_Finalize();
	return 0;
}