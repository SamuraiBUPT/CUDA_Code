#include "add_reduce.h"
#include <cstdlib>
#include <cstdint>
#include "device_functions.h"
#include <iostream>

#ifndef BLOCKDIM
#define BLOCKDIM 128
#endif // !BLOCKDIM

// naive kernel
// ֻ��Ҫ64��kernel����
__global__ void addReduceKernel_v1(int* d_array, unsigned int range) {
	// �������, 0 + 64, 1 + 56, ... 63 + 127
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;	// tid == thread Index

	if (tid > range / 2) return;

	for (int stride = range / 2; stride > 0; stride >>= 1) {
		// ���for���Ǻ��n���̵߳�for��ÿһ��forѭ�����������һ��tier�ļӷ�

		if (tid < stride)
			d_array[tid] += d_array[tid + stride];

		__syncthreads();

	}
}


// unroll - 2: ����block��Ϊһ�����м���
__global__ void addReduceKernel_unroll2(int* d_array, unsigned int range) {
	// ����block�ϲ����㡣֮ǰ��Ҫ64��thread�����ڻ�����Ҫ64����
	uint32_t tid = threadIdx.x;

	// ����������2, 16�ķֲ�
	uint32_t index = blockDim.x * blockIdx.x * 2 + threadIdx.x;	// ���磺0 + 9, 17 + 48... 15 + 47, 
	if (index + blockDim.x < range) {	// �ܷ��ʵ���Զ�ĵط�
		d_array[index] += d_array[index + blockDim.x];
	}

	__syncthreads();

	int* i_data = d_array + blockDim.x * blockIdx.x * 2;
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid < stride)
			i_data[tid] += i_data[tid + stride];	// �������block�ڲ�������ˡ������cpu�������棬���ٴμӺͣ������մ𰸡�
		__syncthreads();
	}

}


void test_add_reduce_main() {
	// NVIDIA GPU: RTX 4070 Laptop
	// SM: 32, 128 cores/SM
	// So let's set the block size to 128
	int original_array[BLOCKDIM];

	std::srand(42);

	for (int i = 0; i < BLOCKDIM; i++) {
		original_array[i] = std::rand();	// ���������������
	}

	int cpu_sum = 0;
	for (int i = 0; i < BLOCKDIM; i++) {
		cpu_sum += original_array[i];
	}

	// ��ʼ��������
	int* d_data;
	cudaMalloc((void**)&d_data, BLOCKDIM * sizeof(int));
	cudaMemcpy(d_data, original_array, BLOCKDIM * sizeof(int), cudaMemcpyHostToDevice);

	addReduceKernel_v1 << <4, 16 >> > (d_data, 128);

	cudaDeviceSynchronize();

	// cpy back
	int new_data[BLOCKDIM] = {0};
	cudaMemcpy(new_data, d_data, BLOCKDIM * sizeof(int), cudaMemcpyDeviceToHost);

	int result_v1 = new_data[0];
	std::cout << "The cpu computing data is: " << cpu_sum << " and the GPU sum data is: " << result_v1 << std::endl;


	// kernel unroll 2
	int* d_data2;
	cudaMalloc((void**)&d_data2, BLOCKDIM * sizeof(int));
	cudaMemcpy(d_data2, original_array, BLOCKDIM * sizeof(int), cudaMemcpyHostToDevice);

	addReduceKernel_unroll2 << <4, 16 >> > (d_data2, 128);

	cudaDeviceSynchronize();

	// cpy back
	int new_data2[BLOCKDIM] = { 0 };
	cudaMemcpy(new_data2, d_data2, BLOCKDIM * sizeof(int), cudaMemcpyDeviceToHost);

	int result_ur2 = 0;
	for (int i = 0; i < 128; i += 16 * 2) {
		result_ur2 += new_data2[i];
	}
	std::cout << "The cpu computing data is: " << cpu_sum << " and the GPU unroll2 sum data is: " << result_ur2 << std::endl;

	cudaFree(d_data);
	cudaFree(d_data2);
	return;
}