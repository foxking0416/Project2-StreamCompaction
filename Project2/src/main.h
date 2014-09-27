#ifndef MAIN_H
#define MAIN_H

#include <stdlib.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include "kernel.h"
#include <ctime>

using namespace std;


int* serialPrefixSumInclusive(int* , int);
int* serialPrefixSumExclusive(int* , int);
int* serialScatter(int* arr, int N);

int main(int argc, char** argv);
void initCuda();
void cleanupCuda();
void shut_down(int return_code);
double diffclock( clock_t clock1, clock_t clock2 );

#endif
