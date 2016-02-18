#include <cstdio>



extern "C" {

__device__
static int THREADS_IN_BLOCK = 1024;


//index - size of left tree

__global__
void init(int* parent, int* tree_size, int* height, bool* computed, int size, int root) {

	int thid = blockIdx.x * blockDim.x + threadIdx.x;	
	if (thid >= size) {
		return;
	}
	if (thid == root) {
		computed[thid] = true;
		parent[thid] = -1;
		height[thid] = 0;

	}
	else {
		computed[thid] = false;
		parent[thid] = root;
	}
	tree_size[thid] = 1;
}

__global__
void quick_sort(int* to_sort, int* parent, int* left, int* right, int* tree_size, int* height, bool* computed, bool* sth_changed, int size) {
	
	int thid = blockIdx.x * blockDim.x + threadIdx.x;	
	if (thid >= size || computed[thid]) {
		return;
	}

	*sth_changed = true;

	//I assume that I have parrent
	int parent_id = parent[thid];
	int my_value = to_sort[thid];
	int parent_value = to_sort[parent_id];


	bool is_left = false;
	if ( my_value < parent_value 
		|| ( my_value == parent_value && thid < parent_id)) {
		is_left = true;
	}

	__syncthreads();
	if (is_left) atomicExch(left + parent_id, thid);
	__syncthreads();

	if (is_left) {
		int left_parent = left[parent_id];
		if (thid == left_parent) {
			computed[thid] = true;
			height[thid] = height[parent_id] + 1;
			// atomicAdd(tree_size + left_parent, 1);  //or init treesaize with 1
		}
		else {
			parent[thid] = left_parent;
			atomicAdd(tree_size + left_parent, 1);
		}
		return;
	}

	__syncthreads();
	atomicExch(right + parent_id, thid);
	__syncthreads();

	int right_parent = right[parent_id];
	if (thid == right_parent) {
		computed[thid] = true;
		// atomicAdd(tree_size + right_parent, 1);  //or init treesaize with 1
	}
	else {
		parent[thid] = right_parent;
		atomicAdd(tree_size + right_parent, 1);
	}
}


}



