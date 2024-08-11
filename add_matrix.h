#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Helper function1
// ��������˺����ġ������˲����Ŀ��������䡢�ͷŵȹ�����
// ͬʱ���㡢����block, thread
cudaError_t addMatrix(float* a, float* b, float* c, uint32_t size);

// C++ API, ֱ�ӵ������
// �ڲ�����������һ������Ȼ����м���
void test_add_matrix();