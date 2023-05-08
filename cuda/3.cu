%%cu
#include <vector>
#include <iostream>
#define N 125
#define M 256
#define BLOCK_DIM 256
#define REDUCE_SIZE (M + BLOCK_DIM - 1) / BLOCK_DIM
#define MAX_GRID_SIZE 256
#define SHMEM_SIZE 4 * BLOCK_DIM

__global__ void reduceRowSum(int* v, int* v_r, int n, int m) {
  __shared__ int partial_sum[BLOCK_DIM][SHMEM_SIZE];

  int tx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;

  if (ty < n && tx < m) {
    partial_sum[threadIdx.y][threadIdx.x] = v[ty * m + tx];
    if (tx + blockDim.x < m) {
        partial_sum[threadIdx.y][threadIdx.x] += v[ty * m * tx + blockDim.x];
    }
  } else {
    partial_sum[threadIdx.y][threadIdx.x] = 0;
  }

  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) 
      partial_sum[threadIdx.y][threadIdx.x] += partial_sum[threadIdx.y][threadIdx.x + s];

    __syncthreads();
  }

  if (threadIdx.x == 0) {
    v_r[ty * m + blockIdx.x] = partial_sum[threadIdx.y][0];
  }
}

int main(void)
{
  int matrix[N][M];
  int btresult[N][(M + BLOCK_DIM - 1) / BLOCK_DIM];
  int bytes = N * M * sizeof(int);
  
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      matrix[i][j] = 1;
    }
  }

  int* d_matrix, *d_btresult;
  cudaMalloc((void**)&d_matrix, bytes);
  cudaMalloc((void**)&d_btresult, N * REDUCE_SIZE);

  cudaMemcpy(d_matrix, matrix, bytes, cudaMemcpyHostToDevice);

  reduceRowSum<<<dim3((M + BLOCK_DIM - 1) / BLOCK_DIM / 2, (N + BLOCK_DIM - 1) / BLOCK_DIM), 
                  dim3(BLOCK_DIM, BLOCK_DIM)>>>(d_matrix, d_btresult); 
  reduceRowSum<<<dim3(1, (N + BLOCK_DIM - 1) / BLOCK_DIM), REDUCE_SIZE>>>(d_btresult, d_btresult);

  for (int i = 0; i < N; i++) {
    std::cout << d_btresult[i * M] << std::endl;
  }

  int* d_v_r;
  cudaMalloc((void**)&d_v_r, sizeof(int) * N);
  
  int GRID_DIM = (N + BLOCK_DIM * 2 - 1) / (BLOCK_DIM * 2);
  reduceSum<<<GRID_DIM, BLOCK_DIM>>>(d_c_s, d_v_r, N);
  reduceSum<<<1, GRID_DIM>>>(d_v_r, d_v_r, GRID_DIM);

  int result;
  cudaMemcpy(&result, d_v_r, sizeof(int), cudaMemcpyDeviceToHost);

  std::cout << result << std::endl;

  // cudaFree(d_a);
  // cudaFree(d_b);
  // cudaFree(d_c_s);
  // cudaFree(d_v_r);
}