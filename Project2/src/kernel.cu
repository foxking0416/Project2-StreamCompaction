#include <stdio.h>
#include <cuda.h>
#include <cmath>
//#include "glm/glm.hpp"
#include "utilities.h"
#include "kernel.h"
#include <thrust/scan.h>

//#include "cuPrintf.cu"


//GLOBALS
dim3 threadsPerBlock(blockSize);

int numObjects;


void checkCUDAError(const char *msg, int line = -1)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        if( line >= 0 ) {
            fprintf(stderr, "Line %d: ", line);
        }
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
        exit(EXIT_FAILURE); 
    }
} 



__global__ void naiveParallelScan (const int *a, int *b, int size, int cut)
{
	int index = threadIdx.x;

	if(index < size){
		if(index >= cut){
			b[index] = a[index - cut] + a[index];
		}
		else{
			b[index] = a[index];
		}
	}
}

__global__ void sharedMemoryParallelScan (const int *a, int *b, int size, int cut){
	__shared__ int temp[blockSize];
	int index = threadIdx.x;

	temp[index] = a[index];
	__syncthreads();

	if(index < size){
		if(index >= cut){
			b[index] = temp[index - cut] + temp[index];
		}
		else{
			b[index] = temp[index];
		}
		__syncthreads();
	}
}

__global__ void sharedMemoryParallelScan2 (const int *a, int *b, int size){
	__shared__ int temp[blockSize];
	__shared__ int temp2[blockSize];


	int index = threadIdx.x;

	temp[index] = a[index];
	__syncthreads();

	for(int offset = 1; offset <= size; offset *= 2){
		//int cut = (int)pow((float)2, (d - 1));

		if(index >= offset){
			temp2[index] = temp[index - offset] + temp[index];
		}
		else{
			temp2[index] = temp[index];
		}
		__syncthreads();
		
		temp[index] = temp2[index];
	}
	b[index] = temp[index];

}

__global__ void sharedMemoryParallelScanArbritraryLength (const int *a, int *b, int size, int cut){
	__shared__ int temp[blockSize];
	//__shared__ int temp2[blockSize];

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;


	temp[threadIdx.x] = a[index];
	__syncthreads();

	if(index < size){
		if(index >= cut){
			b[index] = a[index - cut] + temp[threadIdx.x];
		}
		else{
			b[index] = temp[threadIdx.x];
		}
		__syncthreads();
	}
}

__global__ void scan(float *g_odata, float *g_idata, int n) 
{ 
	 extern __shared__ float temp[]; // allocated on invocation 
	 int thid = threadIdx.x; 
	 int pout = 0, pin = 1; 
	 // load input into shared memory. 
	 // This is exclusive scan, so shift right by one and set first elt to 0 
	 temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0; 
	 __syncthreads(); 
	 for (int offset = 1; offset < n; offset *= 2) 
	 { 
		 pout = 1 - pout; // swap double buffer indices 
		 pin = 1 - pout; 
		 if (thid >= offset) 
			temp[pout*n+thid] += temp[pin*n+thid - offset]; 
		 else 
			temp[pout*n+thid] = temp[pin*n+thid]; 
		 __syncthreads(); 
	 } 
	 g_odata[thid] = temp[pout*n + thid]; // write output 
} 

__global__ void parallelScatterStep1(int *g_idata, int *g_odata, int size) {
	
	__shared__ int temp[blockSize];
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	temp[threadIdx.x] = g_idata[index];
	__syncthreads();
	if(index < size){
		if(temp[threadIdx.x] == 0)
			g_odata[index] = 0;
		else 
			g_odata[index] = 1;

		__syncthreads();
	}
}

__global__ void parallelScatterStep2(int *g_idata, int *postScanData, int *g_odata, int n) {
	__shared__ int temp[blockSize];
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		
	temp[threadIdx.x] = g_idata[index];
	__syncthreads();

	int i = threadIdx.x;

	if(g_idata[i] != 0){
		g_odata[postScanData[i-1]] = g_idata[i];
	}
}


void naiveParallelScan(const int *a, int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *temp = 0;

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));
	cudaMalloc((void**)&temp, size * sizeof(int));

	// Copy input vectors from host memory to GPU buffers.
	cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);

	//clock_t begin = clock();

	//Naive Parallel Scan

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord( start, 0);


	for(int d = 1; d <= (int)ceil(log2((float) size)); ++d){
		int cut = (int)pow((float)2, (d - 1));
		naiveParallelScan <<<1, size>>>(dev_a, dev_b, size, cut);
		dev_a = dev_b;
		cudaMemcpy(dev_a, dev_b, size * sizeof(int), cudaMemcpyDeviceToDevice);
	}
	
	// stop the timer
	cudaEventRecord( stop, 0);
    cudaEventSynchronize( stop );
	float time = 0.0f;
    cudaEventElapsedTime( &time, start, stop);


	printf("Elapsed Time For GPU: %.4f \n", time);

	// Check for any errors launching the kernel
    cudaGetLastError();

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	cudaDeviceSynchronize();

	// Copy output vector from GPU buffer to host memory.
    cudaMemcpy(b, dev_b, size * sizeof(int), cudaMemcpyDeviceToHost);
}

void shareMemoryParallelScan(const int *a, int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *temp = 0;


	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));


	// Copy input vectors from host memory to GPU buffers.
	cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord( start, 0);

	//for(int iter = 0; iter < 1; ++iter){
	//	temp = dev_a;
	//	for(int d = 1; d <= (int)ceil(log2((float) size)); ++d){
	//		int cut = (int)pow((float)2, (d - 1));

	//		sharedMemoryParallelScan <<<1, size>>>(dev_a, dev_b, size, cut);
	//		temp = dev_b;
	//	}
	//}
	//int layerCount = (int)ceil(log2((float) size));
	//cudaMalloc((void**)&temp, layerCount * sizeof(int));

	//int cut = (int)pow((float)2, (d - 1));

	sharedMemoryParallelScan2 <<<1, size>>>(dev_a, dev_b, size);

	// stop the timer
	cudaEventRecord( stop, 0);
    cudaEventSynchronize( stop );
	float time = 0.0f;
    cudaEventElapsedTime( &time, start, stop);
	printf("Elapsed Time For GPU with Share memory: %.4f \n", time);

	// Check for any errors launching the kernel
    cudaGetLastError();

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	cudaDeviceSynchronize();

	// Copy output vector from GPU buffer to host memory.
    cudaMemcpy(b, dev_b, size * sizeof(int), cudaMemcpyDeviceToHost);
}

void shareMemoryParallelScanArbitraryLength(const int *a, int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;

	//cudaError_t cudaStatus;

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));

	// Copy input vectors from host memory to GPU buffers.
	cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);

	dim3 fullBlocksPerGrid((int)ceil(float(size)/float(blockSize)));
	
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord( start, 0);

	for(int d = 1; d <= (int)ceil(log2((float) size)); ++d){
		int cut = (int)pow((float)2, (d - 1));
		sharedMemoryParallelScanArbritraryLength <<<fullBlocksPerGrid, blockSize>>>(dev_a, dev_b, size, cut);
		dev_a = dev_b;
	}

	// stop the timer
	cudaEventRecord( stop, 0);
    cudaEventSynchronize( stop );
	float time = 0.0f;
    cudaEventElapsedTime( &time, start, stop);
	printf("Elapsed Time For GPU with Share memory arbitrary length: %.4f \n", time);

	// Check for any errors launching the kernel
    cudaGetLastError();

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	cudaDeviceSynchronize();

	// Copy output vector from GPU buffer to host memory.
    cudaMemcpy(b, dev_b, size * sizeof(int), cudaMemcpyDeviceToHost);
}


void parallelScatter(const int *a, int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *tempArrPreScan = 0;
	int *tempArrPostScan = 0;

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaMalloc((void**)&dev_a, size * sizeof(int));
	cudaMalloc((void**)&dev_b, size * sizeof(int));
	cudaMalloc((void**)&tempArrPreScan, size * sizeof(int));
	cudaMalloc((void**)&tempArrPostScan, size * sizeof(int));

	// Copy input vectors from host memory to GPU buffers.
	cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);

	dim3 fullBlocksPerGrid((int)ceil(float(size)/float(blockSize)));
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord( start, 0);

	parallelScatterStep1<<<fullBlocksPerGrid, blockSize>>>(dev_a, tempArrPreScan, size);
	for(int d = 1; d <= (int)ceil(log2((float) size)); ++d){
		int cut = (int)pow((float)2, (d - 1));
		//sharedMemoryParallelScan <<<1, size>>>(tempArrPreScan, tempArrPostScan, size, cut);
		sharedMemoryParallelScanArbritraryLength <<<fullBlocksPerGrid, blockSize>>>(tempArrPreScan, tempArrPostScan, size, cut);
		tempArrPreScan = tempArrPostScan;
	}

	//parallelScatterStep2 <<<1, size>>>(dev_a, tempArrPostScan, dev_b, size);

	// stop the timer
	cudaEventRecord( stop, 0);
    cudaEventSynchronize( stop );
	float time = 0.0f;
    cudaEventElapsedTime( &time, start, stop);

	// Check for any errors launching the kernel
    cudaGetLastError();

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	cudaDeviceSynchronize();

	// Copy output vector from GPU buffer to host memory.
    cudaMemcpy(b, tempArrPreScan, size * sizeof(int), cudaMemcpyDeviceToHost);
}



