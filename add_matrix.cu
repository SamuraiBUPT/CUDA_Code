
#include "add_matrix.h"

#include <stdio.h>
#include <windows.h>
#include <cstdlib> // for rand() and srand()
#include <iostream>

#define MAT_SIZE 10

void initialize_mat(float mat[MAT_SIZE][MAT_SIZE]) {
    for (int i = 0; i < MAT_SIZE; i++) {
        for (int j = 0; j < MAT_SIZE; j++) {
            mat[i][j] = rand() % 100;
        }
    }
}

void add_matrix_cpu(float mat1[MAT_SIZE][MAT_SIZE], float mat2[MAT_SIZE][MAT_SIZE], float res[MAT_SIZE][MAT_SIZE]) {
    for (int i = 0; i < MAT_SIZE; i++) {
        for (int j = 0; j < MAT_SIZE; j++) {
            res[i][j] = mat1[i][j] + mat2[i][j];
        }
    }
}

bool valid_matrix(float mat1[MAT_SIZE][MAT_SIZE], float mat2[MAT_SIZE][MAT_SIZE]) {
    for (int i = 0; i < MAT_SIZE; i++) {
        for (int j = 0; j < MAT_SIZE; j++) {
            if (mat1[i][j] != mat2[i][j])
                return false;
        }
    }
    return true;
}

__global__ void addMatrixKernel(float* d_a, float* d_b, float* d_c) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int linear_idx = ix + iy * MAT_SIZE;

    // 一个thread操作10个数字
    int start_idx = linear_idx * MAT_SIZE;
    for (int i = start_idx; i < start_idx + MAT_SIZE; i++) {
        d_c[i] = d_a[i] + d_b[i];
    }
}

cudaError_t addMatrix(float* a, float* b, float* c, uint32_t size) {
    cudaError_t cudaStatus;
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    cudaStatus = cudaMalloc((void**)&d_a, size * sizeof(float));
    cudaStatus = cudaMalloc((void**)&d_b, size * sizeof(float));
    cudaStatus = cudaMalloc((void**)&d_c, size * sizeof(float));

    cudaStatus = cudaMemcpy(d_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(d_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
    // cudaStatus = cudaMemcpy(d_c, c, size * sizeof(float), cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for malloc or memcpy!\n");
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return cudaStatus;
    }

    dim3 grid_dim(2);
    dim3 block_dim(5);

    addMatrixKernel << <grid_dim, block_dim >> > (d_a, d_b, d_c);

    cudaDeviceSynchronize();

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "computing failed!\n");
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return cudaStatus;
}

void test_add_matrix() {
    float a[MAT_SIZE][MAT_SIZE];
    float b[MAT_SIZE][MAT_SIZE];
    float ans[MAT_SIZE][MAT_SIZE];

    // initialize
    initialize_mat(a);
    initialize_mat(b);

    // add by cpu
    add_matrix_cpu(a, b, ans);

    cudaError_t cudaStatus;
    uint32_t matrix_size_linear = MAT_SIZE * MAT_SIZE;
    float c[MAT_SIZE][MAT_SIZE];

    addMatrix((float*)a, (float*)b, (float*)c, matrix_size_linear); // 这里需要做一下类型转换

    bool challenge = valid_matrix(c, ans);
    if (!challenge) {
        printf("Something wrong here.\n");
        return;
    }
    printf("Success!\n");
    return;
}