#ifndef __CUDACC__  
#define __CUDACC__
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <crt/math_functions.h>
#include <crt/device_functions.h>

void print(int* mat, int rows, int cols) {
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			printf("%d ", mat[i * rows + j]);
		}
		printf("\n");
	}
}

__global__ void findMinMatrixDiagonal(int* mat, int n, int* res) {
	__shared__ int shared[256];

	int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
	int gridSize = blockDim.x * gridDim.x * 2;
	int stride = gridSize * (n + 1);

	while (i < n) {
		int mat_id = i * (n + 1);
		shared[threadIdx.x] = min(shared[threadIdx.x], mat[i]);

		if (i + blockDim.x < n) {
			shared[threadIdx.x] = min(shared[threadIdx.x], mat[mat_id + blockDim.x * (n + 1)]);
		}

		i += stride;
	}
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			shared[threadIdx.x] = min(shared[threadIdx.x], shared[threadIdx.x + s]);
		}

		__syncthreads();
	}

	if (threadIdx.x == 0) {
		res[blockIdx.x] = shared[0];
	}
}

__global__ void findMinVector(int* v, int n, int* v_r) {
	__shared__ int shared[256];

	int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
	int gridSize = blockDim.x * gridDim.x * 2;

	while (i < n) {
		shared[threadIdx.x] = min(shared[threadIdx.x], min(v[i], v[i + blockDim.x]));
		i += gridSize;
	}
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		shared[threadIdx.x] = min(shared[threadIdx.x], shared[threadIdx.x + s]);
	}
	__syncthreads();

	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = shared[0];
	}
}

void main() {
	int rows = 8, cols = 8;
	int N = rows * cols;

	int* mat = new int[N] {};
	int* res = new int[N] {};

	for (size_t i = 0; i < N; i++)
	{
		res[i] = INT_MAX;
	}

	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			mat[i * rows + j] = i * rows + j + 1;
		}
	}

	int BLOCK_SIZE = 256;
	int GRID_SIZE = N / BLOCK_SIZE;

	print(mat, rows, cols);

	int* d_mat, * d_res;

	int mallocSize = sizeof(int) * N;

	cudaMalloc(&d_mat, mallocSize);
	cudaMalloc(&d_res, mallocSize);

	cudaMemcpy(d_mat, mat, mallocSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_res, res, mallocSize, cudaMemcpyHostToDevice);

	findMinMatrixDiagonal << <GRID_SIZE / 2, BLOCK_SIZE >> > (d_mat, rows, d_res);
	findMinVector << < 1, BLOCK_SIZE >> > (d_res, rows, d_res);

	cudaMemcpy(res, d_res, mallocSize, cudaMemcpyDeviceToHost);

	printf("\nmin=%d", res[0]);
}
