#include "bank_conflict.h"
#include "device_functions.h"
#include <cstdlib>
#include <cstdint>
#include <iostream>

#define BANK_WIDTH 32	// ��Ϊ32��bank��ÿ��threadһ���պö�Ӧ��

// ���Ǵ���һ��32 * 32�ķ���
// ������������д���������

__global__ void setRowReadRow(int* out) {
	__shared__ int tile[BANK_WIDTH][BANK_WIDTH];

	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	// �������������д��
	tile[threadIdx.y][threadIdx.x] = tid;

	__syncthreads();

	// ������������ж�ȡ
	out[tid] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColReadCol(int* out) {
	__shared__ int tile[BANK_WIDTH][BANK_WIDTH];

	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	// �������������д��
	tile[threadIdx.x][threadIdx.y] = tid;

	__syncthreads();

	// ������������ж�ȡ
	out[tid] = tile[threadIdx.x][threadIdx.y];
}


__global__ void setRowReadColPad(int* out) {
	__shared__ int tile[BANK_WIDTH][BANK_WIDTH + 1];	// �����ôһ�У��ᵼ�´�λ���Ӷ�û��bank conflict

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	tile[threadIdx.y][threadIdx.x] = tid;	// д���ʱ�򣬻���������д�룬���������û��bf

	__syncthreads();

	out[tid] = tile[threadIdx.x][threadIdx.y];	// ��ȡ��ʱ�򣬹����������������᲻����bf
}

void valid_bank_conflict() {
	// 32 * 32 ��linear���ݣ���Ϊʵ�顣
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
		setRowReadColPad << <BANK_WIDTH, BANK_WIDTH >> > (d_out);	// bank conflict�����ʧ�ˣ�����û�㶮ԭ��

		cudaDeviceSynchronize();
	}


	cudaFree(d_out);

	return;
}