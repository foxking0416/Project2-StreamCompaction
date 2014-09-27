#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include <ctime>

using namespace std;

#define blockSize 128
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
#define SHARED 0

void checkCUDAError(const char *msg, int line);
void initCuda(int N);
void naiveParallelScanWithCuda(const int *a, int *b, unsigned int size);
void shareMemoryParallelScanWithCuda(const int *a, int *b, unsigned int size);
void shareMemoryParallelScanArbitraryLengthWithCuda(const int *a, int *b, unsigned int size);
void parallelScatterWithCuda(const int *a, int *b, unsigned int size);
double diffclock( clock_t clock1, clock_t clock2 );



#endif
