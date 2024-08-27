#include "gemm.h"
#include "device_functions.h"
#include <cstdlib>
#include <cstdint>
#include <iostream>

// 假设就是两个方阵在相乘
#define GEMM_M 64
#define GEMM_N 64
#define GEMM_K 64

// the data is linear.
#define WHEREAMI(row, col, ld) ((row) * ld + col)


void initialize_matrix(float* matrix, int rows, int cols) {
	for (int i = 0; i < rows * cols; i++) {
		matrix[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); // 生成0到1之间的随机浮点数
	}
}

bool valid_two_matrices(float* mat1, float* mat2) {
	for (int i = 0; i < GEMM_M; i++) {
		for (int j = 0; j < GEMM_N; j++) {
			if (mat1[WHEREAMI(i, j, GEMM_N)] != mat2[WHEREAMI(i, j, GEMM_N)])
			{
				std::cout << i << " " << j << " has problem." << std::endl;
				std::cout << mat1[WHEREAMI(i, j, GEMM_N)] << " " << mat2[WHEREAMI(i, j, GEMM_N)] << std::endl;
				return false;
			}
		}
	}
	return true;
}

void cpuGEMM(float* a, float* b, float* out,
			int M, int N, int K) {
	// 矩阵运算，说白了每个out的点位，都是由之前的ab矩阵行向量乘列向量出来的
	// a: M行  K列
	// b: K行  N列
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			// 这是内层循环，这个地方代表一个矩阵的一个元素
			float sum = 0.0f;
			for (int tinyk = 0; tinyk < K; tinyk++) {
				sum += a[WHEREAMI(i, tinyk, K)] * b[WHEREAMI(tinyk, j, N)];
			}
			out[WHEREAMI(i, j, N)] = sum;
		}
	}
}

__global__ void gemm_kernel_naive(float* d_a, float* d_b, float* d_out,
								int M, int N, int K) {
	// 最简单的kernel，肯定是每个thread处理一个out的元素
	// RTX 4070 Laptop有 128 cores，一共36个SM，也就是一共有4608个 cores
	// 那就设置 32 block，128 threads
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// int overflow_x = threadIdx.x / K;	// 0, 1
	int current_x = tid / 64;
	int current_y = tid % 64;

	float sum = 0;
	for (int i = 0; i < K; i++) {
		// 这个循环主要就是计算两个向量的乘法的
		sum += d_a[WHEREAMI(current_x, i, K)] * d_b[WHEREAMI(i, current_y, N)];
	}
	d_out[WHEREAMI(current_x, current_y, N)] = sum;
}

void test_gemm_main() {
	srand(static_cast<unsigned>(time(0)));

	// malloc
	float* a = new float[GEMM_M * GEMM_K];
	float* b = new float[GEMM_K * GEMM_N];
	float* cpu_c = new float[GEMM_M * GEMM_N];

	// init
	initialize_matrix(a, GEMM_M, GEMM_K);
	initialize_matrix(b, GEMM_K, GEMM_N);

	cpuGEMM(a, b, cpu_c, GEMM_M, GEMM_N, GEMM_K);

	// malloc
	float* d_a;
	float* d_b;
	float* d_c;
	cudaMalloc((void**)&d_a, GEMM_M * GEMM_K * sizeof(float));
	cudaMalloc((void**)&d_b, GEMM_N * GEMM_K * sizeof(float));
	cudaMalloc((void**)&d_c, GEMM_M * GEMM_N * sizeof(float));

	float* gemm_naive_c = new float[GEMM_M * GEMM_N];

	// cpy
	cudaMemcpy(d_a, a, GEMM_M * GEMM_K * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, GEMM_N * GEMM_K * sizeof(float), cudaMemcpyHostToDevice);

	// launch naive gemm kernel
	gemm_kernel_naive<<<32, 128>>>(d_a, d_b, d_c, GEMM_M, GEMM_N, GEMM_K);

	cudaDeviceSynchronize();

	cudaMemcpy(gemm_naive_c, d_c, GEMM_M * GEMM_N * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << "Naive kernel results: " << valid_two_matrices(cpu_c, gemm_naive_c) << std::endl;


	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	delete[] a;
	delete[] b;
	delete[] cpu_c;
	delete[] gemm_naive_c;

	return;
}