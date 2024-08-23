#include "add_reduce.h"
#include <cstdlib>
#include <cstdint>
#include "device_functions.h"
#include <iostream>

#ifndef BLOCKDIM
#define BLOCKDIM 128
#endif // !BLOCKDIM

// naive kernel
// 只需要64个kernel即可
__global__ void addReduceKernel_v1(int* d_array, unsigned int range) {
	// 交错配对, 0 + 64, 1 + 56, ... 63 + 127
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;	// tid == thread Index

	if (tid > range / 2) return;

	for (int stride = range / 2; stride > 0; stride >>= 1) {
		// 这个for，是横跨n个线程的for，每一次for循环都是在完成一个tier的加法

		if (tid < stride)
			d_array[tid] += d_array[tid + stride];

		__syncthreads();

	}
}


// unroll - 2: 两个block合为一个进行计算
__global__ void addReduceKernel_unroll2(int* d_array, unsigned int range) {
	// 两个block合并计算。之前需要64个thread，现在还是需要64个。
	uint32_t tid = threadIdx.x;

	// 假设现在是2, 16的分布
	uint32_t index = blockDim.x * blockIdx.x * 2 + threadIdx.x;	// 比如：0 + 9, 17 + 48... 15 + 47, 
	if (index + blockDim.x < range) {	// 能访问的最远的地方
		d_array[index] += d_array[index + blockDim.x];
	}

	__syncthreads();

	int* i_data = d_array + blockDim.x * blockIdx.x * 2;
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid < stride)
			i_data[tid] += i_data[tid + stride];	// 在这里，是block内部在相加了。最后在cpu代码里面，会再次加和，求最终答案。
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
		original_array[i] = std::rand();	// 随机生成整数数组
	}

	int cpu_sum = 0;
	for (int i = 0; i < BLOCKDIM; i++) {
		cpu_sum += original_array[i];
	}

	// 开始传输数据
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