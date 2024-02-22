#include <stdio.h>

#define TILE_SIZE 32

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    __shared__ float A_sh[TILE_SIZE][TILE_SIZE];
    __shared__ float B_sh[TILE_SIZE][TILE_SIZE];

    float Cvalue = 0.0;

    for (int t = 0; t < (k - 1) / TILE_SIZE + 1; t++) {
        int xBounds = t * TILE_SIZE + tx;
        if (row < m && xBounds < k) {
            A_sh[ty][tx] = A[row * k + xBounds];
        } else {
            A_sh[ty][tx] = 0.0;
        }
        int yBounds = t * TILE_SIZE + ty;
        if (yBounds < k && col < n) {
            B_sh[ty][tx] = B[yBounds * n + col];
        } else {
            B_sh[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            Cvalue += A_sh[ty][i] * B_sh[i][tx];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = Cvalue;
    }
        
    /*************************************************************************/
}

void basicSgemm(int m, int n, int k, const float *A, const float *B, float *C)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;
	
    /*************************************************************************/
    //INSERT CODE HERE
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n - 1) / BLOCK_SIZE + 1, (m - 1) / BLOCK_SIZE + 1);

    /*************************************************************************/

    // Invoke CUDA kernel -----------------------------------------------------

    /*************************************************************************/
    //INSERT CODE HERE
	mysgemm<<<dimGrid, dimBlock>>>(m, n, k, A, B, C);
    /*************************************************************************/
}


