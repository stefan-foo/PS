%%cu
#include <vector>
#include <iostream>
#define N 15
#define M 300
#define BLOCK_DIM 32
#define REDUCE_SIZE (M + BLOCK_DIM - 1) / BLOCK_DIM

__global__ void reduceRowSum(int* v, int* v_r, int n, int m) {
  __shared__ int partial_sum[BLOCK_DIM][BLOCK_DIM];

  int tx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;

  if (ty < n && tx < m) {
    partial_sum[threadIdx.y][threadIdx.x] = v[ty * m + tx];
    if (tx + blockDim.x < m) {
        partial_sum[threadIdx.y][threadIdx.x] += v[ty * m + tx + blockDim.x];
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
  int btresult[N][M];
  int bytes = N * M * sizeof(int);
  
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      matrix[i][j] = i % 2;
    }
  }

  int* d_matrix, *d_btresult;
  cudaMalloc((void**)&d_matrix, bytes);
  cudaMalloc((void**)&d_btresult, bytes);

  cudaMemcpy(d_matrix, matrix, bytes, cudaMemcpyHostToDevice);
  reduceRowSum<<<dim3((M + BLOCK_DIM * 2 - 1) / (BLOCK_DIM * 2), (N + BLOCK_DIM - 1) / BLOCK_DIM), 
                  dim3(BLOCK_DIM, BLOCK_DIM)>>>(d_matrix, d_btresult, N, M); 
  reduceRowSum<<<dim3(1, (N + BLOCK_DIM - 1) / BLOCK_DIM), dim3((M + BLOCK_DIM * 2 - 1) / (BLOCK_DIM * 2), BLOCK_DIM)>>>(d_btresult, d_btresult, N, M);

  cudaMemcpy(btresult, d_btresult, bytes, cudaMemcpyDeviceToHost);

  int result[N];
  for (int i = 0; i < N; i++) {
    result[i] = btresult[i][0];
    std::cout << i << " " << btresult[i][0] << std::endl;
  }


  // cudaFree(d_a);
  // cudaFree(d_b);
  // cudaFree(d_c_s);
  // cudaFree(d_v_r);
}