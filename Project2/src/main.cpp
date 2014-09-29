#include "main.h"




int main(int argc, char** argv)
{
	///////////////////////////
	//Prepare Array
	///////////////////////////
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


	///////////////////////////
	//CPU Inclusive Prefix Sum
	///////////////////////////
	int* prefixSumInclusiveResult = new int[arraySizeShort];
	clock_t begin = clock();
	for(int iter = 0; iter < iterNum; iter++){
		prefixSumInclusiveResult = serialPrefixSumInclusive(inputArrayShort, arraySizeShort);
	}
	// stop the timer
    clock_t end = clock();
	float time = diffclock(end, begin);
	printf("Elapsed Time For CPU Inclusive Scan: \n(Array length: %d, Iterate times: %d)  %d ms \n", arraySizeShort, iterNum, (int)time);

	printf("Serial Prefix Sum Inclusive last 5 numbers:\n");
	for(int i = 0; i < 5; ++i){
		printf("%d  ", prefixSumInclusiveResult[arraySizeShort - 5 + i]);
	}
	printf("\n\n");
	delete []prefixSumInclusiveResult;


	///////////////////////////
	//CPU Exclusive Prefix Sum
	///////////////////////////
	int* prefixSumExclusiveResult = new int[arraySizeShort];
	begin = clock();
	for(int iter = 0; iter < iterNum; iter++){
		prefixSumExclusiveResult =	serialPrefixSumExclusive(inputArrayShort, arraySizeShort);
	}
	end = clock();
	time = diffclock(end, begin);
	printf("Elapsed Time For CPU Exclusive Scan: \n(Array length: %d, Iterate times: %d) %d ms \n", arraySizeShort, iterNum, (int)time);

	printf("Serial Prefix Sum Exclusive last 5 numbers:\n");
	for(int i = 0; i < 5; ++i){
		printf("%d  ", prefixSumExclusiveResult[arraySizeShort - 5 + i]);
	}
	printf("\n\n");
	delete []prefixSumExclusiveResult;


	///////////////////////////
	//CPU Scatter
	///////////////////////////

	int* scatterResult = new int[arraySizeShort];
	begin = clock();
	for(int iter = 0; iter < iterNum; iter++){
		scatterResult = serialScatter(inputArrayShort, arraySizeShort);
	}
	end = clock();
	time = diffclock(end, begin);
	printf("Elapsed Time For CPU Scatter \n(Array length: %d, Iterate times: %d) %d ms \n", arraySizeShort, iterNum, (int)time);

	printf("Serial Scatter last 5 numbers:\n");
	for(int i = 0; i < 5; ++i){
		printf("%d  ", scatterResult[arraySizeShort / 2 - 5 + i]);
	}
	printf("\n\n");
	delete []scatterResult;

	printf("\n\n\n\n\n");
	printf("*****************Below is GPU version*****************");
	printf("\n\n");


	////////////////////GPU Version///////////////////////////////

	int* resultNaiveScan = new int[arraySizeShort];
	naiveParallelScan(inputArrayShort, resultNaiveScan, arraySizeShort);
	printf("Parallel Naive Scan last 5 numbers:\n");
	for(int i = 0; i < 5; ++i){
		printf("%d  ", resultNaiveScan[arraySizeShort - 5 + i]);
	}
	printf("\n\n");
	delete []resultNaiveScan;

	
	int* resultShareMemoryScan = new int[arraySizeShort];
	shareMemoryParallelScan(inputArrayShort, resultShareMemoryScan, arraySizeShort);
	printf("Parallel Scan Share Memory last 5 numbers:\n");
	for(int i = 0; i < 5; ++i){
		printf("%d  ", resultShareMemoryScan[arraySizeShort - 5 + i]);
	}
	printf("\n\n");
	delete []resultShareMemoryScan;

	//arraySizeLong
	int* resultShareMemoryScanArbLength = new int[arraySizeLong];
	shareMemoryParallelScanArbitraryLength(inputArrayLong, resultShareMemoryScanArbLength, arraySizeLong);
	printf("Parallel Scan Share Memory Arbitrary Length last 5 numbers:\n");
	for(int i = 0; i < 5; ++i){
		//printf("%d  ", resultShareMemoryScanArbLength[128 - 10 +i]);
		printf("%d  ", resultShareMemoryScanArbLength[arraySizeLong - 5 + i]);
	}
	printf("\n\n");
	delete []resultShareMemoryScanArbLength;
	
	
	int* resultScatter = new int[arraySizeLong];
	parallelScatter(inputArrayLong, resultScatter, arraySizeLong);
	printf("Parallel Scatter last 5 numbers:\n");
	for(int i = 0; i < 5; ++i){
		//printf("%d  ", resultScatter[i]);
		//printf("%d  ", resultScatter[128 - 10 +i]);
		printf("%d  ", resultScatter[arraySizeLong / 2 - 5 + i]);
	}
	printf("\n\n");
	delete []resultScatter;


	scatterByThrust(inputArrayLong, arraySizeLong);

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

void scatterByThrust(int* arr, int N){

	size_t n = N;

	thrust::host_vector<int> h_input(n);
	thrust::host_vector<int> h_input_bool(n);
	thrust::host_vector<int> h_map(n);
	thrust::host_vector<int> h_output(n);
	for(int i = 0; i < N ; ++i){
		h_input[i] = arr[i];
	}


	clock_t begin = clock();
	for(int iter = 0; iter < iterNum; ++iter){
		//size_t outputCount = 0;

		for(size_t i = 0; i < n; ++i){
			if(h_input[i] != 0){
				h_input_bool[i] = 1;
				//++outputCount;
			}
		}

		thrust::exclusive_scan(h_input_bool.begin(), h_input_bool.end(), h_map.begin());


		thrust::scatter(h_input.begin(), h_input.end(), h_map.begin(), h_output.begin());
	}
	clock_t end = clock();
	float time = diffclock(end, begin);
	printf("Elapsed Time For Thrust Scatter \n(Array length: %d, Iterate times: %d): %.4f ms \n", N, iterNum, time);
	printf("Thrust Scatter last 5 numbers:\n");
	for(int i = 0; i < 5 ; ++i){
		cout << h_output[h_output.size()/2 - 5 + i] << ", ";
	}
	cout<<endl;

}