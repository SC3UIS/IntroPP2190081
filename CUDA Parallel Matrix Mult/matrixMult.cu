#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

// Global variables used to define number of iterations and matrix values
#define T 10
#define MAX_VAL 1000
#define MIN_VAL 1
#define MAX_DIM 2000

__global__ void matrixMulGPU( int * a, int * b, int * c , int dim)
{
  int val = 0;

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < dim && col < dim)
  {
    for ( int k = 0; k < dim; ++k )
      val += a[row * dim + k] * b[k * dim + col];
    c[row * dim + col] = val;
  }
  
}

void matrixMulCPU( int * a, int * b, int * c, int dim)
{
  int val = 0;

  for( int row = 0; row < dim; ++row )
    for( int col = 0; col < dim; ++col )
    {
      val = 0;
      for ( int k = 0; k < dim; ++k )
        val += a[row * dim + k] * b[k * dim + col];
      c[row * dim + col] = val;
    }
}

int main()
{
  // Structure to measure elapsed time during execution
  struct timeval t0, t1;
  
  int *a, *b, *c_cpu, *c_gpu, dim;

  int size = MAX_DIM * MAX_DIM * sizeof (int); // Number of bytes of an N x N matrix

  // Allocating memory using CUDA directives
  cudaMallocManaged (&a, size);
  cudaMallocManaged (&b, size);
  cudaMallocManaged (&c_cpu, size);
  cudaMallocManaged (&c_gpu, size);
  
  printf("--------------------------------------------\n");
  printf("  CUDA Matrix Multiplication Parallelizing\n");
  printf("--------------------------------------------\n");

  printf("\n");
  printf("--------------------------------------------\n");
  printf("    Size:  Sequential   vs   Parallel \n");
  printf("--------------------------------------------\n");
  
  /*
    There will be a total of T Matrix Multiplications
    its dimension will be increasing in ith powers of 2
    until the Tth power
    
    Times of execution will be measured and displayed
    if and only if there are no discrepancies in the
    resulting matrix
   */
  for (int i = 1; i <= T; i++){
    dim = pow(2, i);
    
    srandom(time(0)+clock()+random());
    for( int row = 0; row < dim; ++row )
      for( int col = 0; col < dim; ++col )
      {
        a[row*dim + col] = rand() % MAX_VAL + MIN_VAL;
        b[row*dim + col] = rand() % MAX_VAL + MIN_VAL;
        c_cpu[row*dim + col] = 0;
        c_gpu[row*dim + col] = 0;
      }
    
    // A 16 x 16 block threads
    dim3 threads_per_block (16, 16, 1); 
    dim3 number_of_blocks ((dim / threads_per_block.x) + 1, (dim / threads_per_block.y) + 1, 1);

    gettimeofday(&t0, 0);
    
    matrixMulGPU <<< number_of_blocks, threads_per_block >>> ( a, b, c_gpu, dim );

    // Directive used to wait until the asynchronous execution finishes, treating it as synchronous
    cudaDeviceSynchronize(); 
    
    gettimeofday(&t1, 0);
    double res_gpu = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;
    
    gettimeofday(&t0, 0);
    // Call the CPU version to check our work
    matrixMulCPU( a, b, c_cpu, dim );
    
    gettimeofday(&t1, 0);
    double res_cpu = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;
    
    // Compare the two answers to make sure they are equal
    bool error = false;
    for( int row = 0; row < dim && !error; ++row )
      for( int col = 0; col < dim && !error; ++col )
        if (c_cpu[row * dim + col] != c_gpu[row * dim + col])
      {
          printf("FOUND ERROR at c[%d][%d]\n", row, col);
          error = true;
          break;
        }
    if (!error)
      printf("%d:    %f                    %f\n",dim, res_cpu, res_gpu  );
        
  }

  // Free all our allocated memory 
  cudaFree(a); cudaFree(b);
  cudaFree( c_cpu ); cudaFree( c_gpu );
  
  
}
