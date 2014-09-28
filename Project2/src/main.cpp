#include "main.h"

#define arraySizeLong 3000
#define arraySizeShort 100

int main(int argc, char** argv)
{


	int *inputArrayLong = new int[arraySizeLong];
	int *inputArrayShort = new int[arraySizeShort];
	for(int i = 0; i < arraySizeLong; ++i){
		if(i % 2 == 0)
			inputArrayLong[i] = 0;
		else
			inputArrayLong[i] = i;
	}
	for(int i = 0; i < arraySizeShort; ++i){
		if(i % 2 == 0)
			inputArrayShort[i] = 0;
		else
			inputArrayShort[i] = i;
	}

	clock_t begin = clock();
	int* prefixSumInclusiveResult = new int[arraySizeLong];
	for(int iter = 0; iter < 100; iter++){
		prefixSumInclusiveResult = serialPrefixSumInclusive(inputArrayLong, arraySizeLong);
	}
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


	int* scatterResult = serialScatter(inputArrayLong, arraySizeLong);
	printf("Serial Scatter\n");
	for(int i = 0; i < 10; ++i){
		printf("%d  ", scatterResult[i]);
	}
	printf("\n\n");
	delete []scatterResult;





	////////////////////GPU Version///////////////////////////////

	int* resultNaiveScan = new int[arraySizeShort];
	naiveParallelScan(inputArrayShort, resultNaiveScan, arraySizeShort);
	printf("Parallel Naive Scan\n");
	for(int i = 0; i < 5; ++i){
		printf("%d  ", resultNaiveScan[arraySizeShort - 5 + i]);
	}
	printf("\n\n");


	
	int* resultShareMemoryScan = new int[arraySizeShort];
	shareMemoryParallelScan(inputArrayShort, resultShareMemoryScan, arraySizeShort);
	printf("Parallel Scan Share Memory\n");
	for(int i = 0; i < 5; ++i){
		printf("%d  ", resultShareMemoryScan[arraySizeShort - 5 + i]);
	}
	printf("\n\n");

	//arraySizeLong
	int* resultShareMemoryScanArbLength = new int[arraySizeLong];
	shareMemoryParallelScanArbitraryLength(inputArrayLong, resultShareMemoryScanArbLength, arraySizeLong);
	printf("Parallel Scan Share Memory Arbitrary Length \n");
	for(int i = 0; i < 5; ++i){
		printf("%d  ", resultShareMemoryScanArbLength[arraySizeLong - 5 + i]);
	}
	printf("\n\n");
	
	
	int* resultScatter = new int[arraySizeLong];
	parallelScatter(inputArrayLong, resultScatter, arraySizeLong);
	printf("Parallel Naive Scatter \n");
	for(int i = 0; i < 10; ++i){
		printf("%d  ", resultScatter[arraySizeShort / 2 - 10 + i]);
		//printf("%d  ", resultScatter[i]);
	}
	printf("\n\n");
	




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

void scanByThrust(){

	//size_t n = 1000;
	//const size_t output_size = std::min((size_t) 10, 2 * n);

	//thrust::host_vector<int> h_input(n);
	//thrust::device_vector<int> d_input(n);
 // 
	//thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

	//for(size_t i = 0; i < n; i++)
	//{
	//	h_map[i] =  h_map[i] % output_size;
	//}

	//thrust::device_vector<unsigned int> d_map = h_map;
 // 
	//thrust::host_vector<int>   h_output(output_size, 0);
	//thrust::device_vector<int> d_output(output_size, 0);


	//thrust::scatter(h_input.begin(), h_input.end(), h_map.begin(), h_output.begin());
}