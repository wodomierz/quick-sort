#include "cuda.h"
#include <cstdio>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <climits> 

#include "quick_sort.h"

static int THREADS_IN_BLOCK = 1024;

using namespace std;
int* quick_sort(int* to_sort, int size){
    cuInit(0);
    CUdevice cuDevice;
    CUresult res = cuDeviceGet(&cuDevice, 0);
    if (res != CUDA_SUCCESS){
        printf("cannot acquire device 0\n");
        exit(1);
    }
    CUcontext cuContext;
    res = cuCtxCreate(&cuContext, 0, cuDevice);
    if (res != CUDA_SUCCESS){
        printf("cannot create Kontext\n");
        exit(1);
    }

    CUmodule cuModule = (CUmodule)0;
    res = cuModuleLoad(&cuModule, "quick_sort.ptx");
    if (res != CUDA_SUCCESS) {
        printf("cannot load module: %d\n", res);
        exit(1);
    }

    CUfunction quick_sort;
    res = cuModuleGetFunction(&quick_sort, cuModule, "quick_sort");
    if (res != CUDA_SUCCESS) printf("some error %d\n", __LINE__);

    CUfunction init;
    res = cuModuleGetFunction(&init, cuModule, "init");
    if (res != CUDA_SUCCESS) printf("some error %d\n", __LINE__);

    int numberOfBlocks = (size+THREADS_IN_BLOCK-1)/THREADS_IN_BLOCK;

    

    int* result = (int*) malloc(sizeof(int) * size);
    cuMemHostRegister((void*) result, size*sizeof(int), 0);
    cuMemHostRegister((void*) to_sort, size*sizeof(int), 0);

    bool* changed;
    cuMemAllocHost((void**) &changed, sizeof(bool));

    CUdeviceptr deviceToSort;
    CUdeviceptr height;
    CUdeviceptr tree_size;
    CUdeviceptr parent;
    CUdeviceptr left;
    CUdeviceptr right;
    CUdeviceptr computed;
    CUdeviceptr sth_changed;

    cuMemAlloc(&deviceToSort, size*sizeof(int));
    cuMemAlloc(&height, size*sizeof(int));
    cuMemAlloc(&tree_size, size*sizeof(int));
    cuMemAlloc(&parent, size*sizeof(int));
    cuMemAlloc(&left, size*sizeof(int));
    cuMemAlloc(&right, size*sizeof(int));
    cuMemAlloc(&computed, size*sizeof(bool));
    cuMemAlloc(&sth_changed, sizeof(bool));

    cuMemcpyHtoD(deviceToSort, (void*) to_sort, size * sizeof(int));


    int root = (rand() % size);
    void* init_args[6] =  { &parent, &tree_size, &height, &computed, &size, &root};
    void* sort_args[9] =  { &deviceToSort, &parent, &left, &right, &tree_size, &height, &computed, &sth_changed, &size};


    cuLaunchKernel(init, numberOfBlocks, 1, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, init_args, 0);
    cuCtxSynchronize();

    // int n;
    // //fit n to power of 2
    // for (n = 1; n<size; n<<=1);

    
    while (true) {
        
        *changed = false;
        cuMemcpyHtoD(sth_changed, (void*) changed, sizeof(bool));

        cuLaunchKernel(quick_sort, numberOfBlocks, 1, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, init_args, 0);
        cuCtxSynchronize();

        cuMemcpyDtoH((void*) changed, sth_changed, sizeof(bool));
        if (not changed) {
           break;
        }
    }

    


   
    
    cuMemcpyDtoH((void*)result, deviceToSort, size * sizeof(int));

    cuMemFree(deviceToSort);
    cuMemHostUnregister(result);
    cuMemHostUnregister(to_sort);
    cuCtxDestroy(cuContext);
    return result;
}
