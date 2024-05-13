#ifndef __DEVICEFUNCTIONS__ 
#define __DEVICEFUNCTIONS__

#include "KDtree.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

// device functions

__global__ void SearchFunction(node* deviceKDtreeArray, float * deviceDistancesArray, subtree_node* deviceSubtreesInformations,int fictitiousNodesDepth,int fictitiousNodesNumber);
__device__ void SortDist(float *deviceDistancesArray, float dist, int id);
__device__ void SharedDepthFirstSearch(int root,int query, float* deviceDistancesArray, node *sharedSubtreeArray, int *index, subtree_node* deviceSubtreesInformations, int blockIdx);
__device__ void DepthFirstSearch(int root,int query, float* deviceDistancesArray, node *deviceKDtreeArray,int *index,int fictitiousNodesDepth, int fictitiousNodesNumber);
__device__ float distance(int *a, int *b);

#endif


