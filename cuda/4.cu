%%cu
#include <vector>
#include <iostream>
#define N 125
#define BLOCK_DIM 256
#define MAX_GRID_SIZE 256
#define SHMEM_SIZE 4 * BLOCK_DIM

__global__ void componentProduct(int* a, int* b, int* out, int len) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  while (tid < len) {
    out[tid] = a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;
  }
}

__global__ void reduceSum(int* v, int* v_r, int len) {
  __shared__ int partial_sum[SHMEM_SIZE];

  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

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

int main(void)
{
  std::vector<int> a(N);
  std::vector<int> b(N);
  
  for (int i = 0; i < N; i++) {
    a[i] = 1;
    b[i] = i + 1;
  }

  int* d_a, *d_b, *d_c_s;
  cudaMalloc((void**)&d_a, sizeof(int) * N);
  cudaMalloc((void**)&d_b, sizeof(int) * N);
  cudaMalloc((void**)&d_c_s, sizeof(int) * N);

  cudaMemcpy(d_a, a.data(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), sizeof(int) * N, cudaMemcpyHostToDevice);

  componentProduct<<<min(MAX_GRID_SIZE, (N + BLOCK_DIM - 1) / BLOCK_DIM), BLOCK_DIM>>>(d_a, d_b, d_c_s, N); 

  int* d_v_r;
  cudaMalloc((void**)&d_v_r, sizeof(int) * N);
 
  int GRID_DIM = (N + BLOCK_DIM * 2 - 1) / (BLOCK_DIM * 2);
  reduceSum<<<GRID_DIM, BLOCK_DIM>>>(d_c_s, d_v_r, N);
  reduceSum<<<1, GRID_DIM>>>(d_v_r, d_v_r, GRID_DIM);

  int result;
  cudaMemcpy(&result, d_v_r, sizeof(int), cudaMemcpyDeviceToHost);

  std::cout << result << std::endl;

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c_s);
  cudaFree(d_v_r);
}