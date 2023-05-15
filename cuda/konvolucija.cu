#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <vector>
#include <iostream>

#define N 2048 * 2048
#define K_S 5
#define TILE_SIZE 32
#define KERNEL_RADIUS 2

__global__ void filterArr(float *in, float *k, float *out, int len, int k_size)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float s_kernel[K_S];

  if (tid < K_S)
  {
    s_kernel[tid] = k[tid];
  }

  __syncthreads();

  float sum = 0;
  int start_point = tid - (k_size / 2);
  for (int i = 0; i < k_size; i++)
  {
    if ((start_point + i) >= 0 && (start_point + i) < len)
    {
      sum += in[tid] * s_kernel[i];
    }
  }

  out[tid] = sum;
}

__global__ void filterOpt(float *in, float *k, float *out, int len, int k_size)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float s_kernel[K_S];
  __shared__ float s_in[TILE_SIZE];

  s_in[threadIdx.x] = in[tid];
  if (threadIdx.x < K_S)
  {
    s_kernel[threadIdx.x] = k[threadIdx.x];
  }

  __syncthreads();

  int stencil = k_size / 2;
  int tile_start_point = blockIdx.x * blockDim.x;
  int tile_end_point = (blockIdx.x + 1) * blockDim.x;
  int start_point = tid - stencil;
  float sum = 0;

  for (int i = 0; i < k_size; i++)
  {
    int c_index = start_point + i;
    if (c_index >= 0 && c_index < len)
    {
      if (c_index >= tile_start_point && c_index < tile_end_point)
      {
        sum += s_in[threadIdx.x - stencil + i] * s_kernel[i];
      }
      else
      {
        sum += in[tid] * s_kernel[i];
      }
    }
  }

  out[tid] = sum;
}

__global__ void filterFullSh(float *in, float *k, float *out, int len, int k_size)
{
  __shared__ float sh[TILE_SIZE + KERNEL_RADIUS * 2];

  int gi = blockIdx.x * blockDim.x + threadIdx.x;
  int li = threadIdx.x + KERNEL_RADIUS;

  sh[li] = gi >= 0 ? in[gi] : 0;

  if (threadIdx.x < KERNEL_RADIUS)
  {
    int loffloc = gi - KERNEL_RADIUS;
    int roffloc = gi + TILE_SIZE;

    sh[li - KERNEL_RADIUS] = loffloc < 0 ? 0 : in[loffloc];
    sh[li + TILE_SIZE] = roffloc >= N ? 0 : in[roffloc];
  }

  __syncthreads();

  float sum = 0;
  for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)
  {
    sum += sh[li + i];
  }

  out[gi] = sum;
}
int main(void)
{
  std::vector<float> in_v(N);
  std::vector<float> result(N);
  std::vector<float> kernel(K_S);

  for (int i = 0; i < N; in_v[i++] = 1)
    ;
  for (int i = 0; i < K_S; kernel[i++] = 1)
    ;

  float *d_a, *d_k, *d_r;
  cudaMalloc((void **)&d_a, sizeof(float) * N);
  cudaMalloc((void **)&d_r, sizeof(float) * N);
  cudaMalloc((void **)&d_k, sizeof(float) * K_S);

  cudaMemcpy(d_a, in_v.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, kernel.data(), sizeof(float) * K_S, cudaMemcpyHostToDevice);

  filterFullSh<<<(N + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE>>>(d_a, d_k, d_r, N, K_S);

  cudaMemcpy(result.data(), d_r, sizeof(float) * N, cudaMemcpyDeviceToHost);

  // for (int i = 0; i < N; i++) {
  // std::cout << result[i] << "\n";
  // }
  std::cout << std::endl;

  cudaFree(d_a);
  cudaFree(d_r);
  cudaFree(d_k);
}