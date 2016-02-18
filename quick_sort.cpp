#include "cuda.h"
#include <cstdio>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <climits> 

#include "quick_sort.h"

static int THREADS_IN_BLOCK = 1024;

using namespace std;

void print_error(CUresult res);

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
    print_error(cuModuleGetFunction(&quick_sort, cuModule, "quick_sort"));

    CUfunction init;
    print_error(cuModuleGetFunction(&init, cuModule, "init"));

    CUfunction tree_to_array;
    print_error(cuModuleGetFunction(&tree_to_array, cuModule, "tree_to_array"));

    int numberOfBlocks = (size+THREADS_IN_BLOCK-1)/THREADS_IN_BLOCK;

    

    int* result = (int*) malloc(sizeof(int) * size);
    cuMemHostRegister((void*) result, size*sizeof(int), 0);
    cuMemHostRegister((void*) to_sort, size*sizeof(int), 0);

    CUdeviceptr device_to_sort;
    CUdeviceptr height;
    CUdeviceptr tree_size;
    CUdeviceptr parent;
    CUdeviceptr left;
    CUdeviceptr right;
    CUdeviceptr computed;
    CUdeviceptr sth_changed;
    CUdeviceptr result_array;
    CUdeviceptr indexes;

    cuMemAlloc(&device_to_sort, size*sizeof(int));
    cuMemAlloc(&height, size*sizeof(int));
    cuMemAlloc(&tree_size, size*sizeof(int));
    cuMemAlloc(&parent, size*sizeof(int));
    cuMemAlloc(&left, size*sizeof(int));
    cuMemAlloc(&right, size*sizeof(int));
    cuMemAlloc(&computed, size*sizeof(bool));
    cuMemAlloc(&sth_changed, sizeof(bool));
    cuMemAlloc(&result_array, size*sizeof(int));
    cuMemAlloc(&indexes, size*sizeof(int));

    print_error(cuMemcpyHtoD(device_to_sort, (void*) to_sort, size * sizeof(int)));


    int root = (rand() % size);
    printf("root %d\n", root);

    void* init_args[8] =  { &parent, &left, &right, &tree_size, &height, &computed, &size, &root};
    void* sort_args[9] =  { &device_to_sort, &parent, &left, &right, &tree_size, &height, &computed, &sth_changed, &size};


    print_error(cuLaunchKernel(init, numberOfBlocks, 1, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, init_args, 0));
    print_error(cuCtxSynchronize());

    // int n;
    // //fit n to power of 2
    // for (n = 1; n<size; n<<=1);

    int counter = 0;
    while (true) {
        // printf("hello %d\n", counter++);
        bool changed = false;
        print_error(cuMemcpyHtoD(sth_changed, &changed, sizeof(bool)));

        print_error(cuLaunchKernel(quick_sort, numberOfBlocks, 1, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, sort_args, 0));
        print_error(cuCtxSynchronize());

        print_error(cuMemcpyDtoH(&changed, sth_changed, sizeof(bool)));
        if (not changed) {
           break;
        }
    }
    int h = 0;
    void* tree_to_array_args[10] = { &device_to_sort, &result_array, &indexes, &parent, &left, &tree_size, &height, &h, &sth_changed, &size};
    while (true) {
        bool changed = false;
        print_error(cuMemcpyHtoD(sth_changed, &changed, sizeof(bool)));

        print_error(cuLaunchKernel(tree_to_array, numberOfBlocks, 1, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, tree_to_array_args, 0));
        print_error(cuCtxSynchronize());
        
        print_error(cuMemcpyDtoH(&changed, sth_changed, sizeof(bool)));
        if (not changed) {
           break;
        }
        h++;
    }

   
    
    print_error(cuMemcpyDtoH((void*)result, result_array, size * sizeof(int)));


    cuMemFree(device_to_sort);
    cuMemFree(height);
    cuMemFree(tree_size);
    cuMemFree(parent);
    cuMemFree(left);
    cuMemFree(right);
    cuMemFree(computed);
    cuMemFree(sth_changed);
    cuMemFree(result_array);
    cuMemFree(indexes);

    cuMemHostUnregister(result);
    cuMemHostUnregister(to_sort);
    cuCtxDestroy(cuContext);
    return result;
}

void print_error(CUresult res) {
    if (res != CUDA_SUCCESS) printf("some error %d\n", __LINE__);
}