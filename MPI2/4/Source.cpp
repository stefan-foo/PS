#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <time.h>
#include <format>
#include <string>

constexpr int N = 5;
constexpr int M = 4;
constexpr int MASTER = 0;

int main(int argc, char** argv)
{
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int dims[2] = { N, M };
	int periods[2] = { true, false };
	int up, down, cart_rank, sum;
	MPI_Comm cart_comm;

	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, true, &cart_comm);
	MPI_Cart_shift(cart_comm, 0, 1, &up, &down);

	sum = up + down;
	std::cout << std::format("WR[{0}] UP[{1}] DOWN[{2}] SUM[{3}]", rank, up, down, sum) << std::endl;

	int total_sum;
	MPI_Reduce(&sum, &total_sum, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);

	if (rank == MASTER) {
		std::cout << "TOTAL SUM: " << total_sum << std::endl;
	}

	MPI_Finalize();
	return 0;
}