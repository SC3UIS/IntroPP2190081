# Parallelizing Matrix Multiplication using CUDA

## Explanation

The purpose of this solution was to test and display the differences in execution time
between the sequential calculation and GPU parallel calculation of the matrix multiplication
algorithm:

$$c_{ij} = \sum_{k = 1}^{m} a_{ik}b_{kj} $$

Matrices were flattened to further optimize the calculations, and to accomodate to CUDA'S
indexing, as described here:
<p align="center">
<img src="./img/cuda_indexing.png" alt="CUDA indexing" style="width:60%;">
    <em><br>CUDA indexing. (2017, January 25). developer.nvidia.com. https://developer.nvidia.com/blog/even-easier-introduction-cuda/</em>
</p> 


Given the a, b, c_cpu and c_gpu flattened square matrices with a maximum dimension of 2000 integers

The calculations in CUDA were done in 16x16 thread blocks with a different number of blocks depending
on the dimension of the matrices to be calculated

Therefore instead of being accessed and calculated sequentially from start to end, the calculations 
were done in small block sized batches each accessed by its its corresponding thread, asynchronously.


## Compiling and executing locally

Run the following while on this directory:

```nvcc matrixMult.cu -arch=sm_70 -o matrixMult -run```
    
    
