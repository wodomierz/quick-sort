#if !defined(BITONIC_SORT_H)
#define BITONIC_SORT_H 1

int* quick_sort_bitonic_merge(int*, int);
void quick_sort(CUdeviceptr device_to_sort, int size, CUdeviceptr result_array);

#endif