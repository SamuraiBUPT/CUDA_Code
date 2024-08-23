# CUDA code
主要用于存放我学习CUDA时候的代码。起因是因为面试HPC岗位的时候，发现自己对CUDA的了解还是很浅薄的，所以决定重新学习一下CUDA。

目前已经实现的kernel有：

- add_matrix: 矩阵加法kernel
- add_reduce: 归约加法kernel，具体的优化有：使用归约方法（相邻归约、交叉配对归约），并且初步接触unroll思想。

# profile工具

使用nsight compute进行profile。

+ Nsight sys主要是针对整个程序进行profile。
+ Nsight compute主要是针对kernel进行profile，包括可以看每个kernel的运行情况、bank conflict等等。


在我们这个场景里面，就用nsight compute进行profile。

```bash
ncu -o profile --set full --section MemoryWorkloadAnalysis_Tables Cuda_Tutorial.exe

ncu -o profile  --metrics group:memory__shared_table Cuda_Tutorial.exe
```

有关更多ncu指令，参考：https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html?highlight=bank%2520conflict#shared-memory

会看到：

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

这种输出。