%%cu
#include <vector>
#include <iostream>
#define N 125
#define BLOCK_DIM 256
#define MAX_GRID_SIZE 256
#define SHMEM_SIZE 4 * BLOCK_DIM

__device__ int minimum(int a, int b) {
  if (a < b) return a;
  else return b;
}

__global__ void findSmallest(int* v, int* v_r, int len) {
  __shared__ int partial_sum[SHMEM_SIZE];

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid < len) {
    partial_sum[threadIdx.x] = v[tid];
    if (partial_sum[threadIdx.x] < 0) {
      partial_sum[threadIdx.x] = INT_MAX;
    }
  } else {
    partial_sum[threadIdx.x] = INT_MAX;
  }

  __syncthreads();
  
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) 
      partial_sum[threadIdx.x] = minimum(partial_sum[threadIdx.x], partial_sum[threadIdx.x + s]);

    __syncthreads();
  }

  if (threadIdx.x == 0) {
    v_r[blockIdx.x] = partial_sum[0];
  }
}

__global__ void swapLZ(int* in, int len, int value) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  while (tid < len) {
    if (in[tid] < 0) {
      in[tid] = value;
    }

    tid += blockDim.x * gridDim.x;
  }
}

int main(void)
{
  std::vector<int> in(N);
  
  for (int i = 0; i < N; i++) {
    if (rand() % 5) {
      in[i] = -rand() % 1000;
    } else {
      in[i] = rand() % 1000;
    }
    std::cout << in[i] << " ";
  }

  int* d_in, *d_out;
  cudaMalloc((void**)&d_in, sizeof(int) * N);
  cudaMalloc((void**)&d_out, sizeof(int) * N);

  cudaMemcpy(d_in, in.data(), sizeof(int) * N, cudaMemcpyHostToDevice);

  int GRID_DIM = (N + BLOCK_DIM * 2 - 1) / (BLOCK_DIM * 2);
  findSmallest<<<GRID_DIM, BLOCK_DIM>>>(d_in, d_out, N);
  findSmallest<<<1,  GRID_DIM>>>(d_out, d_out, GRID_DIM);

  int smallestPositive;
  cudaMemcpy(&smallestPositive, d_out, sizeof(int), cudaMemcpyDeviceToHost);

  swapLZ<<<(N + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>(d_in, N, smallestPositive);
 
  cudaMemcpy(in.data(), d_in, sizeof(int) * N, cudaMemcpyDeviceToHost);

  std::cout << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << in[i] << " ";
  }

  cudaFree(d_in);
  cudaFree(d_out);
}