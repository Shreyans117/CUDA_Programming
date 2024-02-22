#include <stdio.h>

#define TILE_SIZE 16

__global__ void matAdd(int dim, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A + B
     *   where A is a (dim x dim) matrix
     *   where B is a (dim x dim) matrix
     *   where C is a (dim x dim) matrix
     *
     ********************************************************************/

    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < dim && j < dim) {
        int index = j * dim + i;
	    C[index] = A[index] + B[index];
	}
    /*************************************************************************/

}

void basicMatAdd(int dim, const float *A, const float *B, float *C)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;
	
    /*************************************************************************/
    //INSERT CODE HERE
    int noBlocks = (dim + BLOCK_SIZE - 1)/BLOCK_SIZE;
    dim3 DimGrid(noBlocks, noBlocks, 1);
    dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    /*************************************************************************/
	
	// Invoke CUDA kernel -----------------------------------------------------

    /*************************************************************************/
    //INSERT CODE HERE
	matAdd<<<DimGrid, DimBlock>>>(dim, A, B, C);
    /*************************************************************************/

}
		
