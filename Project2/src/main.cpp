#include "main.h"

#define arraySizeLong 10000
#define arraySizeShort 100


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
	int* prefixSumInclusiveResult = new int[arraySizeLong];
	clock_t begin = clock();
	for(int iter = 0; iter < iterNum; iter++){
		prefixSumInclusiveResult = serialPrefixSumInclusive(inputArrayLong, arraySizeLong);
	}
	// stop the timer
    clock_t end = clock();
	float time = diffclock(end, begin);
	printf("Elapsed Time For CPU Inclusive Scan %d iter: %.2f ms \n", iterNum, time);

	printf("Serial Prefix Sum Inclusive\n");
	for(int i = 0; i < 5; ++i){
		printf("%d  ", prefixSumInclusiveResult[arraySizeLong - 5 + i]);
	}
	printf("\n\n");
	delete []prefixSumInclusiveResult;


	///////////////////////////
	//CPU Exclusive Prefix Sum
	///////////////////////////
	int* prefixSumExclusiveResult = new int[arraySizeLong];
	begin = clock();
	for(int iter = 0; iter < iterNum; iter++){
		prefixSumExclusiveResult =	serialPrefixSumExclusive(inputArrayLong, arraySizeLong);
	}
	end = clock();
	time = diffclock(end, begin);
	printf("Elapsed Time For CPU Exclusive Scan %d iter: %.4f ms \n",iterNum, time);

	printf("Serial Prefix Sum Exclusive\n");
	for(int i = 0; i < 5; ++i){
		printf("%d  ", prefixSumExclusiveResult[arraySizeLong - 5 + i]);
	}
	printf("\n\n");
	delete []prefixSumExclusiveResult;


	///////////////////////////
	//CPU Scatter
	///////////////////////////

	int* scatterResult = new int[arraySizeLong];
	begin = clock();
	for(int iter = 0; iter < iterNum; iter++){
		scatterResult = serialScatter(inputArrayLong, arraySizeLong);
	}
	end = clock();
	time = diffclock(end, begin);
	printf("Elapsed Time For CPU Scatter %d iter: %.4f ms \n", iterNum, time);

	printf("Serial Scatter\n");
	for(int i = 0; i < 10; ++i){
		printf("%d  ", scatterResult[arraySizeLong / 2 - 10 + i]);
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
		//printf("%d  ", resultShareMemoryScanArbLength[128 - 10 +i]);
		printf("%d  ", resultShareMemoryScanArbLength[arraySizeLong - 5 + i]);
	}
	printf("\n\n");
	
	
	int* resultScatter = new int[arraySizeLong];
	parallelScatter(inputArrayLong, resultScatter, arraySizeLong);
	printf("Parallel Naive Scatter \n");
	for(int i = 0; i < 10; ++i){
		printf("%d  ", resultScatter[i]);
		//printf("%d  ", resultScatter[128 - 10 +i]);
		//printf("%d  ", resultScatter[arraySizeLong / 2 - 15 + i]);
	}
	printf("\n\n");
	


	scanByThrust(inputArrayLong, arraySizeLong);



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

void scanByThrust(int* arr, int N){

	//thrust::host_vector<int> H(4);
	//H[0] = 14;
	//H[1] = 20;
	//H[2] = 38;
	//H[3] = 46;


	//std::cout << "H has size " << H.size() << std::endl;
	//for(int i = 0; i < H.size(); i++)
	//	std::cout << "H[" << i << "] = " << H[i] << std::endl;

	//H.resize(2);
	//std::cout << "H now has size " << H.size() << std::endl;

	//thrust::device_vector<int> D = H;


	size_t n = 10;
	//const size_t output_size = std::min((size_t) 10, 2 * n);

	thrust::host_vector<int> h_input(n);
	h_input[0] = 0;
	h_input[1] = 1;
	h_input[2] = 0;
	h_input[3] = 3;
	h_input[4] = 0;
	h_input[5] = 5;
	h_input[6] = 0;
	h_input[7] = 7;
	h_input[8] = 0;
	h_input[9] = 9;

	//h_input[0] = 1;
	//h_input[1] = 0;
	//h_input[2] = 3;
	//h_input[3] = 0;
	//h_input[4] = 5;
	//h_input[5] = 0;
	//h_input[6] = 1;
	//h_input[7] = 0;
	//h_input[8] = 1;
	//h_input[9] = 0;
	cout<< "Size is " << h_input.size() << endl;


	cout << "h_input :" << endl;
	for(int i = 0; i < h_input.size() ; ++i){
		cout << h_input[i] << ", ";
	}
	cout<<endl;


	size_t outputCount = 0;
	thrust::host_vector<int> h_input_bool(n);
	for(size_t i = 0; i < n; ++i){
		if(h_input[i] != 0){
			h_input_bool[i] = 1;
			++outputCount;
		}
	}

	thrust::host_vector<int> h_map(n);
	thrust::exclusive_scan(h_input_bool.begin(), h_input_bool.end(), h_map.begin());

	cout << "h_map :" << endl;
	for(int i = 0; i < h_map.size() ; ++i){
		cout << h_map[i] << ", ";
	}
	cout<<endl;


	//thrust::sequence(X.begin(), X.end());
	//for(int i = 0; i < X.size(); i++)
	//	std::cout << "X[" << i << "] = " << X[i] << std::endl;

	//thrust::host_vector<unsigned int> h_map(n);
	//h_map[0] = 0;
	//h_map[1] = 0;
	//h_map[2] = 1;
	//h_map[3] = 1;
	//h_map[4] = 2;
	//h_map[5] = 2;
	//h_map[6] = 3;
	//h_map[7] = 3;
	//h_map[8] = 4;
	//h_map[9] = 4;



	//for(size_t i = 0; i < n; i++)
	//{
	//	h_map[i] =  h_map[i] % output_size;
	//}

	//thrust::device_vector<unsigned int> d_map = h_map;
 // 
	//thrust::host_vector<int>   h_output(output_size, 0);
	//thrust::device_vector<int> d_output(output_size, 0);


	thrust::host_vector<int>   h_output(outputCount);
	thrust::scatter(h_input.begin(), h_input.end(), h_map.begin(), h_output.begin());
	for(int i = 0; i < h_output.size() ; ++i){
		cout << h_output[i] << ", ";
	}
	cout<<endl;

}