#ifndef __HOSTFUNCTIONS__ 
#define __HOSTFUNCTIONS__

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <ctime>

#define N 2
#define K 5
#define NUMBER_OF_POINTS 10000
#define MAX_SUBTREE_DIMENSION 1024




typedef struct tree_node{   int coord[N];
							int name; // può essere tolto
							int dad;
							int right_child;
							int left_child;
						    int depth; }node;

typedef struct tree_coord{   int coord[N];  }coord;

typedef struct subtree{  int startNode;
                         int  subtreeElements;    }subtree_node;

// host functions

void hostDeallocateRam(node* hostKDtreeArray, subtree_node *hostSubtreesInformations, float * hostDistancesArray,coord *datasetCoordinatesArray,int *fictitiousNodesIndexArray);
int buildKDtree(node *hostKDtreeArray,coord *datasetCoordinatesArray, int* index, int myindex, int depth,  int start, int stop, int fictitiousNodesDepth,int *fictitiousNodesIndexArray, int* j);
void mergeSort(coord* arr,int low,int mid,int high, int Axis);
void partition(coord* arr,int low,int high, int Axis);
void findSubtrees (int fictitiousNodesDepth,int fictitiousNodesNumber, node* hostKDtreeArray,int index , int *count, subtree_node* hostSubtreesInformations);

#endif