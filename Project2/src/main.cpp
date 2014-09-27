#include "main.h"

#define arraySizeLong 3000
#define arraySizeShort 100

int main(int argc, char** argv)
{

	int initial[10] = {0, 0, 3, 4, 0, 6, 6, 7, 0, 1};
	int *inputArrayLong = new int[arraySizeLong];
	int *inputArrayShort = new int[arraySizeShort];
	for(int i = 0; i < arraySizeLong; ++i){
		inputArrayLong[i] = i;
	}
	for(int i = 0; i < arraySizeShort; ++i){
		inputArrayShort[i] = i;
	}

	clock_t begin = clock();
	int* prefixSumInclusiveResult = serialPrefixSumInclusive(inputArrayLong, arraySizeLong);
	// stop the timer
    clock_t end = clock();
	float time = diffclock(end, begin);
	printf("Elapsed Time For CPU : %.8f \n", time);

	printf("Serial Prefix Sum Inclusive\n");
	for(int i = 0; i < 5; ++i){
		printf("%d  ", prefixSumInclusiveResult[arraySizeLong - 5 + i]);
	}
	printf("\n\n");
	delete []prefixSumInclusiveResult;


	int* prefixSumExclusiveResult = serialPrefixSumExclusive(inputArrayLong, arraySizeLong);
	printf("Serial Prefix Sum Exclusive\n");
	for(int i = 0; i < 5; ++i){
		printf("%d  ", prefixSumExclusiveResult[arraySizeLong - 5 + i]);
	}
	printf("\n\n");
	delete []prefixSumExclusiveResult;


	int* scatterResult = serialScatter(initial, 10);
	printf("Serial Scatter\n");
	for(int i = 0; i < 10; ++i){
		printf("%d  ", scatterResult[i]);
	}
	printf("\n\n");
	delete []scatterResult;





	////////////////////GPU Version///////////////////////////////



	int* resultNaiveScan = new int[arraySizeShort];
	naiveParallelScanWithCuda(inputArrayShort, resultNaiveScan, arraySizeShort);
	printf("Parallel Naive Scan\n");
	for(int i = 0; i < 5; ++i){
		printf("%d  ", resultNaiveScan[arraySizeShort - 5 + i]);
	}
	printf("\n\n");


	
	int* resultShareMemoryScan = new int[arraySizeShort];
	shareMemoryParallelScanWithCuda(inputArrayShort, resultShareMemoryScan, arraySizeShort);
	printf("Parallel Scan Share Memory\n");
	for(int i = 0; i < 5; ++i){
		printf("%d  ", resultShareMemoryScan[arraySizeShort - 5 + i]);
	}
	printf("\n\n");

	//arraySizeLong
	int* resultShareMemoryScanArbLength = new int[arraySizeLong];
	shareMemoryParallelScanArbitraryLengthWithCuda(inputArrayLong, resultShareMemoryScanArbLength, arraySizeLong);
	printf("Parallel Scan Share Memory Arbitrary Length \n");
	for(int i = 0; i < 5; ++i){
		printf("%d  ", resultShareMemoryScanArbLength[arraySizeLong - 5 + i]);
	}
	//for(int i = 0; i < arraySize; ++i){
	//	printf("%d  ", resultShareMemoryScanArbLength[i]);
	//}
	printf("\n\n");
	
	/*
	int* resultScatter = new int[arraySize];
	parallelScatterWithCuda(initial, resultScatter, 10);
	printf("Parallel Naive Scatter \n");
	for(int i = 0; i < 10; ++i){
		printf("%d  ", resultScatter[i]);
	}
	printf("\n\n");
	*/




	int stop = 0;
	scanf("%d", &stop);

    return 0;
}


void shut_down(int return_code)
{
    exit(return_code);
}


int* serialPrefixSumInclusive(int* arr, int N){

	int* result = new int[N];
	for(int i = 0; i < N; ++i){
		if(i == 0)
			result[i] = arr[i];
		else
			result[i] = result[i -1] + arr[i];
	}
	return result;
}

int* serialPrefixSumExclusive(int* arr, int N){

	int* result = new int[N];
	for(int i = 0; i < N; ++i){
		if(i == 0)
			result[i] = 0;
		else
			result[i] = result[i -1] + arr[i - 1];
	}
	return result;
}

int* serialScatter(int* arr, int N){
	
	int* firstStep = new int[N];
	int* arrAfterScan = new int[N];
	int lengthCompact = 0;
	for(int i = 0; i < N; ++i){
		if(arr[i] == 0)
			firstStep[i] = 0;
		else{
			firstStep[i] = 1;
			++lengthCompact;
		}
	}

	arrAfterScan = serialPrefixSumExclusive(firstStep, N);

	int* result = new int[lengthCompact];
	for(int i = 0; i < N; ++i){
		if(firstStep[i] == 1){
			result[arrAfterScan[i]] = arr[i];
		}
	}
	return result;
}

double diffclock( clock_t clock1, clock_t clock2 )
{
    double diffticks = clock1 - clock2;
    double diffms    = diffticks / ( CLOCKS_PER_SEC / 1000.0);
    return diffms;
}
