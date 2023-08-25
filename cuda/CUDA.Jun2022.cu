#include <stdio.h>

__global__ void matrixAdd(float *a, float *b, float *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int index = i + j * n;
    if (i < n && j < n) {
        c[index] = a[index] + b[index];
    }
}

__global__ void colAverage(float *matrix, float *avgVec, int n) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += matrix[i + j * n];
    }
    avgVec[j] = sum / n;
}

int main() {
    int n = 1024;
    int size = n * n * sizeof(float);
    float *a, *b, *c, *avgVec;
    cudaMalloc((void **)&a, size);
    cudaMalloc((void **)&b, size);
    cudaMalloc((void **)&c, size);
    cudaMalloc((void **)&avgVec, n * sizeof(float));
    float *hostA, *hostB, *hostC, *hostAvgVec;
    hostA = (float *)malloc(size);
    hostB = (float *)malloc(size);
    hostC = (float *)malloc(size);
    hostAvgVec = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n * n; i++) {
        hostA[i] = (float)i;
        hostB[i] = (float)i;
    }
    cudaMemcpy(a, hostA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b, hostB, size, cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixAdd<<<numBlocks, threadsPerBlock>>>(a, b, c, n);
    colAverage<<<(n + 255) / 256, 256>>>(c, avgVec, n);
    cudaMemcpy(hostC, c, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostAvgVec, avgVec, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n * n; i++) {
        if (i % n == 0) {
            printf("\n");
        }
        printf("%.1f ", hostC[i]);
    }
    printf("\n");
    for (int i = 0; i < n; i++) {
        printf("%.1f ", hostAvgVec[i]);
    }
    printf("\n");
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(avgVec);
    free(hostA);
    free(hostB);
    free(hostC);
    free(hostAvgVec);
    return 0;
}
