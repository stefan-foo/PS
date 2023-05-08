%%cu

// 5. Koristeći CUDA tehnologiju, sastaviti program koji sve elemente matrice A koji su veći od prosečne
// vrednosti u matrici menja brojem -1. Maksimalno redukovati broj pristupa globalnoj memoriji.
// Obratiti pažnju na efikasnost paralelizacije. Omogućiti rad programa za nizove proizvoljne veličine.

#include <vector>
#include <iostream>
#define N 256
#define M 128
#define BLOCK_DIM 256
#define MAX_GRID_SIZE 256
#define SHMEM_SIZE 4 * BLOCK_DIM

__global__ void reduce(int* v, int* v_r, int len) {
  __shared__ int partial_sum[SHMEM_SIZE];

  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < len) {
      partial_sum[threadIdx.x] = v[i];
      if (i + blockDim.x < len) {
          partial_sum[threadIdx.x] += v[i + blockDim.x];
      }
  } else {
      partial_sum[threadIdx.x] = 0;
  }

  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) 
      partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];

    __syncthreads();
  }

  if (threadIdx.x == 0) {
    v_r[blockIdx.x] = partial_sum[0];
  }
}

__global__ void replaceGreaterThan(int* arr, int len, float value) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < len) {
    if (arr[tid] < value) {
      arr[tid] = -1;
    }
  }
}

int main(void)
{
  int matrix[N][M];
  size_t arr_size = N * M;
  size_t bytes = sizeof(int) * arr_size;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      matrix[i][j] = rand()  % 20;
    }
  }

  int* d_matrix, *d_sum;
  cudaMalloc((void**)&d_matrix, bytes);
  cudaMalloc((void**)&d_sum, bytes);

  cudaMemcpy(d_matrix, matrix, bytes, cudaMemcpyHostToDevice);

  int GRID_DIM = (arr_size + BLOCK_DIM * 2 - 1) / (BLOCK_DIM * 2);
  reduce<<<GRID_DIM, BLOCK_DIM>>>(d_matrix, d_sum, arr_size);
  reduce<<<1, GRID_DIM>>>(d_sum, d_sum, GRID_DIM);

  int result;
  cudaMemcpy(&result, d_sum, sizeof(int), cudaMemcpyDeviceToHost);

  replaceGreaterThan<<<GRID_DIM, BLOCK_DIM>>>(d_matrix, arr_size, result * 1.0 / N / M);

  cudaMemcpy(matrix, d_matrix, bytes, cudaMemcpyDeviceToHost);

  std::cout << "AVG " << result * 1.0 / N / M << std::endl;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      std::cout << matrix[i][j] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << result << std::endl;

  cudaFree(d_matrix);
  cudaFree(d_sum);
}