#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <time.h>

constexpr int N = 4;
constexpr int M = 4;

//2. Napisati MPI program kojim se kreira dvodimenzionalna Cartesian struktura sa n vrsta i m
//kolona.U svakom od nxm procesa odštampati identifikatore procesa njegovog levog i desnog
//suseda na udaljenosti 2. Smatrati da su procesi u prvoj i poslednjoj koloni jedne vrste susedni.

int main(int argc, char** argv)
{
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int dims[2] = { N, M };
	int periods[2] = { false, true };
	MPI_Comm cart_comm;

	int left, right;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, true, &cart_comm);
	MPI_Cart_shift(cart_comm, 1, 2, &left, &right);

	std::cout << "RANK[" << rank << "] " << "L[" << left << "] " << "R[" << right << "]" << std::endl;

	MPI_Finalize();
	return 0;
}