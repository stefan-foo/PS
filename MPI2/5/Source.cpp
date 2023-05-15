#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <time.h>

constexpr int M = 7;
constexpr int N = 7;
constexpr int MASTER = 0;

//3. Napisati MPI program kojim se kreira dvodimenzionalna Cartesian struktura sa n vrsta i m
//kolona.U svakom procesu odštampati identifikatore i koordinate njegovog levog i desnog
//suseda na udaljenosti 3. Ilustrovati raspored procesa i diskutovati dobijeno rešenje u zavisnosti
//od periodičnosti dimenzija.Program testirati za različite vrednosti n i m.

int main(int argc, char** argv)
{
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Comm cart_comm;
	// 3 * 8
	int dims[2] = { 8, 3 };
	int periods[2] = { true, true };
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, false, &cart_comm);

	int coords[2];
	MPI_Cart_coords(cart_comm, rank, 2, coords);

	int left, right;
	MPI_Cart_shift(cart_comm, 0, 3, &left, &right);

	std::cout << "WORLD RANK: [" << rank << "] LEFT " << left << " RIGHT " << right << std::endl;

	// 4 * 6
	MPI_Barrier(cart_comm);
	if (rank == 0) {
		std::cout << "size: 4 * 6" << std::endl;
	}
	MPI_Barrier(cart_comm);
	MPI_Comm_free(&cart_comm);

	dims[0] = 6; dims[1] = 4;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, false, &cart_comm);
	MPI_Cart_coords(cart_comm, rank, 2, coords);
	MPI_Cart_shift(cart_comm, 0, 3, &left, &right);

	std::cout << "WORLD RANK: [" << rank << "] LEFT " << left << " RIGHT " << right << std::endl;
	
	// 2 * 12
	MPI_Barrier(cart_comm);
	if (rank == 0) {
		std::cout << "size: 2 * 12" << std::endl;
	}
	MPI_Barrier(cart_comm);
	MPI_Comm_free(&cart_comm);

	dims[0] = 12; dims[1] = 2;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, true, &cart_comm);
	MPI_Cart_coords(cart_comm, rank, 2, coords);
	MPI_Cart_shift(cart_comm, 0, 3, &left, &right);

	std::cout << "WORLD RANK: [" << rank << "] LEFT " << left << " RIGHT " << right << std::endl;

	MPI_Finalize();
	return 0;
}