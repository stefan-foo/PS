%%cu
#include <vector>
#include <iostream>
#define TILE_SIZE 32
#define MAX_GRID_SIZE 256
#define K_S 2

__global__ void map(float* in, float* out, int len) {
  __shared__ float sh[TILE_SIZE + 2];

  int i = threadIdx.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  while (tid < len) {
      sh[i + 2] = in[tid + 2];
      if (i < 2) {
          sh[i] = in[tid];
      }
      __syncthreads();

      out[tid] = (3 * sh[i] + 10 * sh[i+1] + 7 * sh[i+2]) / 20.f;

      __syncthreads();

      tid += blockDim.x * gridDim.x;
  }
}

int main(void)
{
    int n = 272;
    std::vector<float> vec_in(n + 2);
    std::vector<float> vec_out(n);
    
    for (int i = 0; i < n + 2; i++) {
        vec_in[i] = rand() % 20; 
    }

    float* d_in, *d_out;
    cudaMalloc((void**)&d_in, sizeof(float) * (n + 2));
    cudaMalloc((void**)&d_out, sizeof(float) * n);

    cudaMemcpy(d_in, vec_in.data(), sizeof(float) * (n + 2), cudaMemcpyHostToDevice);

    map<<<min(MAX_GRID_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE), TILE_SIZE>>>(d_in, d_out, n); 

    cudaMemcpy(vec_out.data(), d_out, sizeof(float) * n, cudaMemcpyDeviceToHost);
 
    //(3 * sh[i] + 10 * sh[i+1] + 7 * sh[i+2]) / 20.f;
 
    for (int i = 0; i < n; i++) {
        std::cout << vec_out[i] << "-" << ((3 * vec_in[i] + 10 * vec_in[i+1] + 7 * vec_in[i+2]) / 20.f) << " ";
    }

    cudaFree(d_in);
    cudaFree(d_out);
}