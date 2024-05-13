#ifndef __RESULTS_WRITE__ 
#define __RESULTS_WRITE__

#include "Dataset_read.h"

void writeTreeToFile (node* hostKDtreeArray, int Number);
void writeResultsToFile (float* hostDistancesArray,int fictitiousNodesNumber, int* fictitiousNodesIndexArray); 

#endif