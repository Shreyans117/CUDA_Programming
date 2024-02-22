#include <stdio.h>
#define BLOCK_SIZE 512

__global__ void histo_kernel(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins)
{
	
    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ unsigned int histo_private[4096];
    for (int j = threadIdx.x; j < num_bins; j+= blockDim.x ) {
        histo_private[j]=0;
    }
    __syncthreads();
    // stride is total number of threads
    int stride = blockDim.x * gridDim.x;
    // All threads handle blockDim.x * gridDim.x
    // consecutive elements
    while (i < num_elements) {
        atomicAdd(&(histo_private[input[i]]), 1);
        i += stride;
    }

	__syncthreads(); 

    int k = threadIdx.x;
    while (k < num_bins) {
        atomicAdd(&(bins[k]), histo_private[k] );
        k += blockDim.x;
    }
	  /*************************************************************************/
}

void histogram(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) {

	  /*************************************************************************/
          //INSERT CODE HERE

          dim3 blockDim (BLOCK_SIZE, 1, 1);
	  dim3 numBlocks ((num_elements-1)/BLOCK_SIZE + 1, 1, 1);
          histo_kernel<<<numBlocks,blockDim>>>(input, bins, num_elements, num_bins);
	  /*************************************************************************/

}


