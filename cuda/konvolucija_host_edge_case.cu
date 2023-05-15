% % cu
#include <vector>
#include <iostream>
#define N 37
#define TILE_SIZE 256
#define K_S 2

        __global__ void
        map(float *in, float *out, int len, int k_size)
{
  __shared__ float shared[TILE_SIZE + 2 * K_S];

  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int i = threadIdx.x + K_S;

  if (tid < len)
  {
    // shared[i] = in[tid];
    shared[i] = in[tid + k_size];

    if (threadIdx.x < K_S)
    {
      shared[threadIdx.x] = in[tid];
      shared[i + TILE_SIZE] = in[tid + TILE_SIZE + k_size];
    }

    __syncthreads();

    float sum = 0;
    for (int j = -K_S; j <= K_S; j++)
    {
      sum += shared[i + j];
    }

    out[tid] = sum;
  }
}

int main(void)
{
  std::vector<float> in_v(N + 2 * K_S, 0);
  std::vector<float> result(N);

  for (int i = K_S; i < (N + K_S + 1); in_v[i++] = 1)
    ;

  float *d_a, *d_r;
  cudaMalloc((void **)&d_a, sizeof(float) * (N * K_S));
  cudaMalloc((void **)&d_r, sizeof(float) * N);

  cudaMemcpy(d_a, in_v.data(), sizeof(float) * (N * K_S), cudaMemcpyHostToDevice);

  map<<<(N + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE>>>(d_a, d_r, N, K_S);

  cudaMemcpy(result.data(), d_r, sizeof(float) * N, cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++)
  {
    std::cout << result[i] << " ";
  }
  std::cout << std::endl;

  cudaFree(d_a);
  cudaFree(d_r);
}