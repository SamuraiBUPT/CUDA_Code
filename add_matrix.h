#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Helper function1
// 负责包裹核函数的。控制了参数的拷贝、分配、释放等工作。
// 同时计算、分配block, thread
cudaError_t addMatrix(float* a, float* b, float* c, uint32_t size);

// C++ API, 直接调用这个
// 内部会自行生成一个矩阵，然后进行计算
void test_add_matrix();