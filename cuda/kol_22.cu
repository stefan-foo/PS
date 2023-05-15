% % cu
#include <vector>
#include <iostream>
#define BLOCK_SIZE 256
#define MAX_GRID_SIZE 256

        __global__ void
        map(float *a, float *b, float *c, float p, int size)
{
  __shared__ float shared_a[BLOCK_SIZE + 2];
  __shared__ float shared_b[BLOCK_SIZE + 2];

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int i2 = threadIdx.x + 2;
  int i = threadIdx.x;

  while (tid < size - 2)
  {
    shared_a[i2] = a[tid + 2];
    shared_b[i2] = b[tid + 2];

    if (threadIdx.x < 2)
    {
      shared_a[i] = a[tid];
      shared_b[i] = b[tid];
    }

    __syncthreads();
    c[tid] = (shared_a[i] + shared_a[i + 1] + shared_a[i + 2]) * p + (shared_b[i] + shared_b[i + 1] + shared_b[i + 2]) * (1 - p);
    tid += blockDim.x * gridDim.x;
  }
}

int main(void)
{
  int n = 272;
  float p = 0.25;
  std::vector<float> vec_a(n);
  std::vector<float> vec_b(n);
  std::vector<float> vec_c(n - 2);

  for (int i = 0; i < n; i++)
  {
    vec_a[i] = rand() % 20; // 3
    vec_b[i] = rand() % 20;
  }

  float *da, *db, *dc;
  cudaMalloc(&da, sizeof(float) * n);
  cudaMalloc(&db, sizeof(float) * n);
  cudaMalloc(&dc, sizeof(float) * n);

  cudaMemcpy(da, vec_a.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(db, vec_b.data(), sizeof(float) * n, cudaMemcpyHostToDevice);

  map<<<min(MAX_GRID_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(da, db, dc, p, n);

  cudaMemcpy(vec_c.data(), dc, sizeof(float) * (n - 2), cudaMemcpyDeviceToHost);

  for (int i = 0; i < n - 2; i++)
  {
    std::cout << vec_c[i] << "-" << (vec_c[i] == ((vec_a[i] + vec_a[i + 1] + vec_a[i + 2]) * p + (vec_b[i] + vec_b[i + 1] + vec_b[i + 2]) * (1 - p))) << " ";
  }

  cudaFree(db);
  cudaFree(da);
  cudaFree(dc);
}