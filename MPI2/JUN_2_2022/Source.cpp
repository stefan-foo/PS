#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <time.h>

struct Student {
	int ind;
	char ime[20];
	char prezime[20];
};

int main(int argc, char** argv)
{
	int rank, size;

	Student student = {};

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int q = sqrt(size);
	if (q * q != size) return -1;

	int array_of_bl[] = { 1, 20, 20 };
	MPI_Datatype arra_of_types[] = { MPI_INT, MPI_CHAR, MPI_CHAR };
	MPI_Aint array_of_dsp[3];

	MPI_Get_address(&student.ind, &array_of_dsp[0]);
	MPI_Get_address(&student.ime, &array_of_dsp[1]);
	MPI_Get_address(&student.prezime, &array_of_dsp[2]);
	array_of_dsp[1] -= array_of_dsp[0];
	array_of_dsp[2] -= array_of_dsp[0];
	array_of_dsp[0] = 0;

	MPI_Datatype student_type;
	MPI_Type_create_struct(3, array_of_bl, array_of_dsp, arra_of_types, &student_type);
	MPI_Type_commit(&student_type);
	// MPI_UNDEFINED, MPI_COMM_NULL

	MPI_Comm diag;
	MPI_Group world_group, diag_group;

	MPI_Comm_group(MPI_COMM_WORLD, &world_group);
	int *ranks = new int[q];
	for (int i = 0; i < q; i++) {
		ranks[i] = i * (q + 1);
	}
	MPI_Group_incl(world_group, q, ranks, &diag_group);
	MPI_Comm_create(MPI_COMM_WORLD, world_group, &diag);

	if (rank == 0) {
		student.ind = 17975;
		strcpy_s(student.ime, "gyros");
		strcpy_s(student.prezime, "gyros");
	}

	if (diag != MPI_COMM_NULL) {
		MPI_Bcast(&student, 1, student_type, 0, diag);

		std::cout << "RANK: " << rank << "\n";
		std::cout << student.ind << "\n";
		std::cout << student.ime << "\n";
		std::cout << student.prezime << "\n";
	}
	else {
		std::cout << "RANK: " << diag << "\n";
	}

	MPI_Finalize();
	return 0;
}