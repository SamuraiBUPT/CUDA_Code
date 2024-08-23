# CUDA code
主要用于存放我学习CUDA时候的代码。起因是因为面试HPC岗位的时候，发现自己对CUDA的了解还是很浅薄的，所以决定重新学习一下CUDA。

目前已经实现的kernel有：

- add_matrix: 矩阵加法kernel
- add_reduce: 归约加法kernel，具体的优化有：使用归约方法（相邻归约、交叉配对归约），并且初步接触unroll思想。_