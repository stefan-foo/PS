#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <time.h>

//6. Napisati MPI program kojim se kreira dvodimenzionalna Cartesian struktura sa n vrsta i m
//kolona. Za svaki skup procesa koji pripadaju istoj koloni strukture kreirati novi komunikator.
//Master procesu iz svake kolone poslati koordinate procesa sa najvećim identifikatorom i
//prikazati ih

constexpr int N = 5;
constexpr int M = 6;
constexpr int MASTER = 5;

int main(int argc, char** argv)
{
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	const int dims[] = { N, M };
	const int periods[] = { false, false };

 	MPI_Comm cart_comm, col_comm, row_comm;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

	int cart_coords[2];
	MPI_Cart_coords(cart_comm, rank, 2, cart_coords);
	int master_coords[2];
	MPI_Cart_coords(cart_comm, MASTER, 2, master_coords);

	MPI_Comm_split(cart_comm, cart_coords[0], cart_coords[1], &row_comm);
	MPI_Comm_split(cart_comm, cart_coords[1], cart_coords[0], &col_comm);

	int col_rank, max_col_rank;
	MPI_Comm_rank(row_comm, &col_rank);
	MPI_Reduce(&rank, &max_col_rank, 1, MPI_INT, MPI_MAX, master_coords[0], col_comm);

	if (cart_coords[0] == master_coords[0]) {
		//std::cout << "[" << cart_coords[0] << ", " << cart_coords[1] << "] " << col_rank << " " << max_col_rank << std::endl;

		int max_col_ranks[M] = { 0 };

		MPI_Gather(&max_col_rank, 1, MPI_INT, max_col_ranks, 1, MPI_INT, master_coords[1], row_comm);

		if (cart_coords[1] == master_coords[1]) {
			for (int i = 0; i < M; i++) {
				std::cout << max_col_ranks[i] << " ";
			}
		}
	}

	MPI_Finalize();
	return 0;
}