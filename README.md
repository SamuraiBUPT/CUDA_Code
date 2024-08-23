# CUDA code
��Ҫ���ڴ����ѧϰCUDAʱ��Ĵ��롣��������Ϊ����HPC��λ��ʱ�򣬷����Լ���CUDA���˽⻹�Ǻ�ǳ���ģ����Ծ�������ѧϰһ��CUDA��

Ŀǰ�Ѿ�ʵ�ֵ�kernel�У�

- add_matrix: ����ӷ�kernel
- add_reduce: ��Լ�ӷ�kernel��������Ż��У�ʹ�ù�Լ���������ڹ�Լ��������Թ�Լ�������ҳ����Ӵ�unroll˼�롣

# profile����

ʹ��nsight compute����profile��

+ Nsight sys��Ҫ����������������profile��
+ Nsight compute��Ҫ�����kernel����profile���������Կ�ÿ��kernel�����������bank conflict�ȵȡ�


����������������棬����nsight compute����profile��

```bash
ncu -o profile --set full --section MemoryWorkloadAnalysis_Tables Cuda_Tutorial.exe

ncu -o profile  --metrics group:memory__shared_table Cuda_Tutorial.exe
```

�йظ���ncuָ��ο���https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html?highlight=bank%2520conflict#shared-memory

�ῴ����

```
D:\Code_Space\HPC\code-vs\Cuda_Tutorial\x64\Debug>ncu -o profile Cuda_Tutorial.exe
==PROF== Connected to process 4596 (D:\Code_Space\HPC\code-vs\Cuda_Tutorial\x64\Debug\Cuda_Tutorial.exe)
==PROF== Profiling "setRowReadRow(int *)" - 0: 0%....50%....100% - 8 passes
==PROF== Profiling "setRowReadRow(int *)" - 1: 0%....50%....100% - 8 passes
==PROF== Profiling "setRowReadRow(int *)" - 2: 0%....50%....100% - 8 passes
==PROF== Profiling "setRowReadRow(int *)" - 3: 0%....50%....100% - 8 passes
==PROF== Profiling "setRowReadRow(int *)" - 4: 0%....50%....100% - 8 passes
==PROF== Profiling "setColReadCol(int *)" - 5: 0%....50%....100% - 8 passes
==PROF== Profiling "setColReadCol(int *)" - 6: 0%....50%....100% - 8 passes
==PROF== Profiling "setColReadCol(int *)" - 7: 0%....50%....100% - 8 passes
==PROF== Profiling "setColReadCol(int *)" - 8: 0%....50%....100% - 8 passes
==PROF== Profiling "setColReadCol(int *)" - 9: 0%....50%....100% - 8 passes
==PROF== Disconnected from process 4596
==PROF== Report: D:\Code_Space\HPC\code-vs\Cuda_Tutorial\x64\Debug\profile.ncu-rep
```

���������