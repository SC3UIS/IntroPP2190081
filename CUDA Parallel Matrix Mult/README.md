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

```shell
nvcc matrixMult.cu -arch=sm_70 -o matrixMult -run
```

## Code execution in GUANE

Once you are in GUANE you will run the next commands: 

```shell
srun -n 8 --pty /bin/bash
```
In this case we are using a machine with 8 nodes and the maximum GPU quantity available, but, in the case that you will need to require an specific number of GPUs we will use the next command: 
```shell
srun -n 8 --gres=gpu:2 --pty /bin/bash
```

In this specific case, we are going to use 2 GPUs

Before running the code, modifications were made to enable its execution on CUDA 8.0, as the code was originally written in CUDA 11.8.

To run the code, you will use the next command:
```shell
nvcc matrixMult.cu -arch=sm_30 -o matrixMult -run
```
When executing the command, we obtain the following:

<img width="593" alt="image" src="https://github.com/SC3UIS/IntroPP2191621/assets/67378380/3e431832-7d4b-43f1-88c8-7f832e7ca866">
