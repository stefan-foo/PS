#include <stdio.h>

#define N 1024 // size of the vectors
#define THREADS_PER_BLOCK 256 // number of threads per block
#define P 0.5f // the value of p in the formula

__global__ void vectorCalc(float *a, float *b, float *c)
{
    __shared__ float shared_a[THREADS_PER_BLOCK+2]; // add padding to avoid shared memory bank conflicts
    __shared__ float shared_b[THREADS_PER_BLOCK+2]; // add padding to avoid shared memory bank conflicts

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    shared_a[tid] = a[i];
    shared_b[tid] = b[i];

    if (tid == THREADS_PER_BLOCK-1 || tid == THREADS_PER_BLOCK-2)
    {
        shared_a[tid+2] = a[i+2];
        shared_b[tid+2] = b[i+2];
    }

    __syncthreads();    

    c[tid] = (shared_a[tid] + shared_a[tid+1] + shared_a[tid+2]) * P +
             (shared_b[tid] + shared_b[tid+1] + shared_b[tid+2]) * (1-P);
}

int main()
{
    float *a, *b, *c; // input vectors and output vector
    float *d_a, *d_b, *d_c; // device copies of input vectors and output vector

    // Allocate memory for input vectors and output vector on host
    a = (float*)malloc(N*sizeof(float));
    b = (float*)malloc(N*sizeof(float));
    c = (float*)malloc(N*sizeof(float));

    // Initialize input vectors on host
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = 2*i;
    }

    // Allocate memory for input vectors and output vector on device
    cudaMalloc(&d_a, N*sizeof(float));
    cudaMalloc(&d_b, N*sizeof(float));
    cudaMalloc(&d_c, N*sizeof(float));

    // Copy input vectors from host to device
    cudaMemcpy(d_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N*sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel to perform calculation on two vectors
    vectorCalc<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

    // Copy output vector from device to host
    cudaMemcpy(c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the result
    for (int i = 0; i < N; i++)
    {
        printf("%f\n", c[i]);
    }

    // Free memory on host and device
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
