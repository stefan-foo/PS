%%cu
#include <vector>
#include <iostream>
#define N 455
#define M 382
#define TILE_SIZE 256
#define MAX_GRID_SIZE 256


__global__ void maxPerCol(int* in, int* out, int n, int m) {
int tid = threadIdx.x + blockDim.x * blockIdx.x;
int max = -1;

while (tid < m) {
  for (int i = 0; i < n; i++) {
    int cur = in[i * m + tid];
    max = max < cur ? cur : max;
  }

  out[tid] = max;
  tid += blockDim.x * gridDim.x;
}

}
int main(void)
{
    int matrix[N][M];
    std::vector<int> col_max(M);
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
          matrix[i][j] = rand() % 1000;
        }
    }

    int* d_matrix, *d_out;
    cudaMalloc((void**)&d_matrix, sizeof(int) * M * N);
    cudaMalloc((void**)&d_out, sizeof(int) * M);

    cudaMemcpy(d_matrix, matrix, sizeof(int) * M * N, cudaMemcpyHostToDevice);

    maxPerCol<<<min(MAX_GRID_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE), TILE_SIZE>>>(d_matrix, d_out, N, M); 

    cudaMemcpy(col_max.data(), d_out, sizeof(int) * M, cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; i++) {
        std::cout << col_max[i] << " ";
    }

    cudaFree(d_matrix);
    cudaFree(d_out);
}