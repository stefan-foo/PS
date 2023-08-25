#ifndef __CUDACC__  
#define __CUDACC__
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <crt/math_functions.h>
#include <corecrt_math.h>

#define BLOCK_SIZE_X 2
#define GRID_SIZE 128
#define N 20

__global__ void addMatrices(int* a, int* b, int* c) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	while (i < N)
	{
		c[i] = a[i] + b[i];
		i += gridDim.x * blockDim.x;
	}
}

__global__ void minimumEveryRow(int* mat, int* medj) {
	__shared__ int sh[BLOCK_SIZE_X];

	// medj[N][gridDim.x]

	int tx = threadIdx.x + blockDim.x * blockIdx.x * 2;
	int tx_start = tx;
	int ty = blockIdx.y;

	int stride_y = N * gridDim.y;

	while (ty < N) {
		tx = tx_start;
		while (tx < N)
		{
			sh[threadIdx.x] = min(sh[threadIdx.x], mat[ty * N + tx]);

			if (ty* N + tx + blockDim.x < N) {
				sh[threadIdx.x] = min(sh[threadIdx.x], mat[ty * N + tx + blockDim.x]);
			}
			tx += gridDim.x * blockDim.x * 2;
		}
		__syncthreads();

		for (int s = blockDim.x / 2; s > 0; s /= 2)
		{
			if (threadIdx.x < s) {
				sh[threadIdx.x] = min(sh[threadIdx.x], sh[threadIdx.x + s]);
			}
			__syncthreads();
		}

		if (threadIdx.x == 0) {
			medj[ty * N + blockIdx.x] = sh[0];
		}

		ty += stride_y;
	}
}

__global__ void min(int* medj, int* res) { // medj[N][blockDim.x], res[N]
	__shared__ int sh[BLOCK_SIZE_X];

	int ty = blockIdx.y;
	int stride_y = gridDim.y;

	// blockDim.x = prethodni gridDim.x
	// blockDim.y = 1
	// gridDim.x = 1, gridDim.y = proizvoljno

	while (ty < N) {
		sh[threadIdx.x] = medj[ty * N + threadIdx.x];
		__syncthreads();

		for (int s = blockDim.x / 2; s > 0; s /= 2)
		{
			if (threadIdx.x < s) {
				sh[threadIdx.x] = min(sh[threadIdx.x], sh[threadIdx.x + s]);
			}
			__syncthreads();
		}

		if (threadIdx.x == 0) {
			res[ty] = sh[0];
		}

		ty += stride_y;
	}
}

int main()
{
	int* d_a, * d_b, * d_c, * d_medj, * d_res;

#define GRIDDIM_X 2
#define GRIDDIM_Y 2

	int* a = (int*)malloc(N * N * sizeof(int));
	int* b = (int*)malloc(N * N * sizeof(int));
	int* c = (int*)malloc(N * N * sizeof(int));
	int* medj = (int*)malloc(N * GRIDDIM_X * sizeof(int));
	int* res = (int*)malloc(N * sizeof(int));

	cudaMalloc(&d_a, N * N * sizeof(int));
	cudaMalloc(&d_b, N * N * sizeof(int));
	cudaMalloc(&d_c, N * N * sizeof(int));
	cudaMalloc(&d_medj, N * GRIDDIM_X * sizeof(int));
	cudaMalloc(&d_res, N * sizeof(int));

	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			a[i * N + j] = b[i * N + j] = rand() % 2;
		}
	}

	cudaMemcpy(d_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, a, N * N * sizeof(int), cudaMemcpyHostToDevice);

	addMatrices << <GRID_SIZE, BLOCK_SIZE_X >> > (d_a, d_b, d_c);

	cudaMemcpy(c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);


	printf("a:\n");
	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			printf("%d ", a[i * N + j]);
		}
		printf("\n");
	}

	printf("b:\n");
	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			printf("%d ", b[i * N + j]);
		}
		printf("\n");
	}

	printf("c:\n");
	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			printf("%d ", c[i * N + j]);
		}
		printf("\n");
	}

	printf("\n");

	dim3 grid(GRIDDIM_X, GRIDDIM_Y);
	dim3 blk(BLOCK_SIZE_X, 1);
	minimumEveryRow << <grid, blk >> > (d_c, d_medj);

	grid = dim3(1, 2);
	blk = dim3(BLOCK_SIZE_X, 1);
	min << <grid, blk >> > (d_medj, d_res);

	cudaMemcpy(res, d_res, N * sizeof(int), cudaMemcpyDeviceToHost);

	printf("\nv: \n");
	for (size_t i = 0; i < N; i++)
	{
		printf("%d ", res[i]);
	}
	printf("\n");
}
