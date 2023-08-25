#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// Kernel function for box filter
__global__ void boxFilterKernel(const int* inputMatrix, float* outputMatrix, int rows, int columns, int filterSize) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < rows && j < columns) {
		int filterRadius = filterSize / 2;
		float sum = 0;
		int count = 0;

		for (int m = -filterRadius; m <= filterRadius; m++) {
			for (int n = -filterRadius; n <= filterRadius; n++) {
				int row = i + m;
				int col = j + n;

				if (row >= 0 && row < rows && col >= 0 && col < columns) {
					sum += inputMatrix[row * rows + col];
					count++;
				}
			}
		}

		outputMatrix[i * rows + j] = sum / count;
	}
}

void seqBoxFilter(int* input, float* output, int rows, int cols, int filterSize) {
	int filterRadius = filterSize / 2;

	for (int i = 0; i < rows; i++) {
		for (int j = 0;j < cols; j++) {
			int sum = 0;
			int count = 0;
			for (int m = -filterRadius; m <= filterRadius; m++) {
				for (int n = -filterRadius; n <= filterRadius; n++) {
					int row = i + m;
					int col = j + n;

					if (row >= 0 && row < rows && col >= 0 && col < cols) {
						sum += input[row * rows + col];
						count++;
					}
				}
				output[i * rows + j] = 1.0f * sum / count;
			}
		}
	}
}

// Function to perform box filter on matrix using CUDA
void performBoxFilter(const int* inputMatrix, float* outputMatrix, int rows, int columns, int filterSize) {
	int matrixSize = rows * columns;
	int* deviceInputMatrix;
	float* deviceOutputMatrix;

	// Allocate device memory
	cudaMalloc((void**)&deviceInputMatrix, matrixSize * sizeof(int));
	cudaMalloc((void**)&deviceOutputMatrix, matrixSize * sizeof(float));

	// Copy input matrix from host to device
	cudaMemcpy(deviceInputMatrix, inputMatrix, matrixSize * sizeof(int), cudaMemcpyHostToDevice);

	// Define the grid and block dimensions
	dim3 blockSize(16, 16);
	dim3 gridSize((columns + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

	// Launch the kernel
	boxFilterKernel << <gridSize, blockSize >> > (deviceInputMatrix, deviceOutputMatrix, rows, columns, filterSize);

	// Copy the result from device to host
	cudaMemcpy(outputMatrix, deviceOutputMatrix, matrixSize * sizeof(float), cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(deviceInputMatrix);
	cudaFree(deviceOutputMatrix);
}

int main() {
	int rows = 16;
	int columns = 16;
	int filterSize = 3;

	// Generate the input matrix (you can replace this with your own input matrix)
	int* inputMatrix = new int[rows * columns];
	// Fill inputMatrix with your own data

	for (int i = 0;i < rows;i++) {
		for (int j = 0;j < columns;j++) {
			inputMatrix[i * rows + j] = rand() % 2;
		}
	}
	// Print the result (you can modify this to suit your needs)
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			printf("%.2f ", inputMatrix[i * rows + j]);
		}
		std::cout << std::endl;
	}
	printf("\n");

	// Allocate memory for the output matrix
	float* outputMatrix = new float[rows * columns]();
	float* outputMatrixSeq = new float[rows * columns]();

	// Perform the box filter
	performBoxFilter(inputMatrix, outputMatrix, rows, columns, filterSize);

	seqBoxFilter(inputMatrix, outputMatrixSeq, rows, columns, 3);

	// Print the result (you can modify this to suit your needs)
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			if (outputMatrix[i * rows + j] != outputMatrixSeq[i * rows + j]) {
				printf("ERR  ");
			}
			else {

				printf("%.2f ", outputMatrix[i * rows + j]);
			}
		}
		std::cout << std::endl;
	}

	// Free memory
	delete[] inputMatrix;
	delete[] outputMatrix;

	return 0;
}
