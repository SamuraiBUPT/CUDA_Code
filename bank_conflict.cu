#include "bank_conflict.h"
#include "device_functions.h"
#include <cstdlib>
#include <cstdint>
#include <iostream>

#define BANK_WIDTH 32	// 因为32哥bank，每个thread一个刚好对应。

// 我们创建一个32 * 32的方阵。
// 首先是行主序写、行主序读

__global__ void setRowReadRow(int* out) {
	__shared__ int tile[BANK_WIDTH][BANK_WIDTH];

	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	// 按照行主序进行写入
	tile[threadIdx.y][threadIdx.x] = tid;

	__syncthreads();

	// 按照行主序进行读取
	out[tid] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColReadCol(int* out) {
	__shared__ int tile[BANK_WIDTH][BANK_WIDTH];

	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	// 按照列主序进行写入
	tile[threadIdx.x][threadIdx.y] = tid;

	__syncthreads();

	// 按照列主序进行读取
	out[tid] = tile[threadIdx.x][threadIdx.y];
}


__global__ void setRowReadColPad(int* out) {
	__shared__ int tile[BANK_WIDTH][BANK_WIDTH + 1];	// 添加那么一列，会导致错位，从而没有bank conflict

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	tile[threadIdx.y][threadIdx.x] = tid;	// 写入的时候，还是行主序写入，这样本身就没有bf

	__syncthreads();

	out[tid] = tile[threadIdx.x][threadIdx.y];	// 读取的时候，故意这样，看看还会不会有bf
}

void valid_bank_conflict() {
	// 32 * 32 的linear数据，作为实验。
	int origin_arr[BANK_WIDTH * BANK_WIDTH];
	std::srand(42);
	for (int i = 0; i < BANK_WIDTH * BANK_WIDTH; i++) {
		origin_arr[i] = std::rand();
	}

	int* d_out;
	cudaMalloc((void**)&d_out, BANK_WIDTH * BANK_WIDTH * sizeof(int));
	
	for (int i = 0; i < 5; i++) {
		setRowReadRow << <BANK_WIDTH, BANK_WIDTH >> > (d_out);

		cudaDeviceSynchronize();
	}

	for (int i = 0; i < 5; i++) {
		setColReadCol << <BANK_WIDTH, BANK_WIDTH >> > (d_out);

		cudaDeviceSynchronize();
	}

	for (int i = 0; i < 5; i++) {
		setRowReadColPad << <BANK_WIDTH, BANK_WIDTH >> > (d_out);	// bank conflict真的消失了，但是没搞懂原因

		cudaDeviceSynchronize();
	}


	cudaFree(d_out);

	return;
}