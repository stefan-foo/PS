#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <time.h>

constexpr int v = 20;
constexpr int MASTER = 0;
constexpr int N = 6; // pokrenuti za manje od 19 procesa

//3. Napisati MPI program koji kreira komunikator comm1 koji se sastoji od svih procesa sa
// identifikatorima deljivim sa 3. Master proces(P0) svim procesima ove grupe šalje po jednu
// vrstu matrice A.Odštampati identifikatore procesa koji pripadaju comm1 i čija je suma
// elemenata primljene vrste matrice A manja od zadate vrednosti v.

int main(int argc, char** argv)
{
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int k3_rank;
	MPI_Group k3_group, world_group;
	MPI_Comm k3_comm;

	int n = (size + 2) / 3;
	int* incl = new int[n];

	for (int i = 0; i < n; i++) {
		incl[i] = i * 3;
	}

	MPI_Comm_group(MPI_COMM_WORLD, &world_group);

	MPI_Group_incl(world_group, n, incl, &k3_group);
	MPI_Comm_create(MPI_COMM_WORLD, k3_group, &k3_comm);

	if (k3_comm != MPI_COMM_NULL) {
		int A[N][N] = { 0 }, row[N] = { 0 };

		if (rank == MASTER) {
			for (int i = 0; i < N * N; i++) {
				((int*)A)[i] = rand() % 4;
			}
		}

		MPI_Group_rank(k3_group, &k3_rank);
		MPI_Scatter(A, N, MPI_INT, row, N, MPI_INT, MASTER, k3_comm);
		
		int sum = 0;
		for (int i = 0; i < N; i++) {
			sum += row[i];
		}

		if (sum < v) {
			std::cout << "WR[" << rank << "] " << " K3[" << k3_rank << "] SUM: " << sum << std::endl;
		}
	}

	MPI_Finalize();
	return 0;
}