#include "KDtree.h"
#include "Dataset_read.h"
#include "Results_write.h"
#include "CUDA_KDtree.cuh"

cudaError_t knnWithCuda(node *hostKDtreeArray, subtree_node *hostSubtreesInformations, float *hostDistancesArray,int fictitiousNodesDepth, int fictitiousNodesNumber);
void deviceError(node* deviceKDtreeArray, subtree_node *deviceSubtreesInformations, float * deviceDistancesArray);

int main()
{

    cudaProfilerStart(); 

    node *hostKDtreeArray=NULL;
    coord *datasetCoordinatesArray=NULL;
    subtree_node *hostSubtreesInformations=NULL;
    float *hostDistancesArray=NULL;
    int  *fictitiousNodesIndexArray=0;

	int fictitiousNodesDepth, fictitiousNodesNumber;

    /* 
     * given the number of points of the dataset we compute how much fictitious nodes we have to insert in the space, in order to build
     * a KDtree which have a maximum of 1024 valid elements from dataset for each subtree. Of course if the number of point is less 
     * than 1024 no fictitious nodes are added. We compute also the depth until which our KDtree present fictitious nodes. 
     */

    if(NUMBER_OF_POINTS<1024)
        {
            fictitiousNodesDepth=-1;
            fictitiousNodesNumber=0;
        }
    else
        {
            int exponent=(int) logb((float)NUMBER_OF_POINTS) - 9 ;
            fictitiousNodesNumber= (int) pow(2, (double)exponent)-1;
            fictitiousNodesDepth= (int) logb((float)fictitiousNodesNumber);
        }

    /* 
     * we allocate space in host memory for the structures we need for implement the alghoritm :
     * - fictitiousNodesIndexArray stores the index of the fictitious nodes in the KDtreeArray
     * - hostKDtreeArray is the array with which we build the KDtree
     * - datasetCoordinatesArray contains the coordinates of dataset that will be sorted in order to build a balanced KDtree
     * - hostSubtreesInformations stores the index of the roots and the number of elements of each subtrees
     * - hostDistancesArray contains the final results that is the kNN distances
     */

    fictitiousNodesIndexArray=(int*)malloc(sizeof(int)*fictitiousNodesNumber);
    if(fictitiousNodesIndexArray==NULL)
    {
        printf("Malloc failed!\n");
        return -1;
    }
	hostKDtreeArray=(node*)malloc(sizeof(node)*(NUMBER_OF_POINTS+fictitiousNodesNumber));
    if(hostKDtreeArray==NULL)
    {
        printf("Malloc failed!\n");
        return -1;
    }
    datasetCoordinatesArray=(coord*)malloc(sizeof(coord)*NUMBER_OF_POINTS);
    if(datasetCoordinatesArray==NULL)
    {
        printf("Malloc failed!\n");
        return -1;
    }
    hostSubtreesInformations=(subtree_node*)malloc(sizeof(subtree_node)*(fictitiousNodesNumber+1));
    if(hostSubtreesInformations==NULL)
    {
        printf("Malloc failed!\n");
        return -1;
    }
    hostDistancesArray=(float*)malloc(sizeof(float)*K*(NUMBER_OF_POINTS + fictitiousNodesNumber));
    if(hostDistancesArray==NULL)
    {
        printf("Malloc failed!\n");
        return -1;
    }

    // read dataset from file  
    if(readPointsFromFile(datasetCoordinatesArray)==-1)
    return -1;

    //compute time

    float startTime=clock();

    // build the KDtree
    int  i, index=-1, currentIndex=-1,root, depth = 0, fictitiousCount=0, start=0;
    root=buildKDtree(hostKDtreeArray,datasetCoordinatesArray, &index, currentIndex , depth, start , NUMBER_OF_POINTS-1, fictitiousNodesDepth, fictitiousNodesIndexArray, &fictitiousCount);

    // find roots and dimensions of each subtrees
    int count=0;
    index=0;
    findSubtrees(fictitiousNodesDepth,fictitiousNodesNumber, hostKDtreeArray,index, &count , hostSubtreesInformations);

    float stopTime=clock();

    printf("timeElapse :  %f ms\n", ((stopTime - startTime)*1000)/CLOCKS_PER_SEC);

    // print on file KDtree
    writeTreeToFile(hostKDtreeArray,NUMBER_OF_POINTS+fictitiousNodesNumber);

    // initialize the hostDistancesArray to the maximum float
    for(i=0; i<K*(NUMBER_OF_POINTS + fictitiousNodesNumber); i++)
    hostDistancesArray[i]=FLT_MAX;

    // search knn distances in parallel.
    cudaError_t cudaStatus = knnWithCuda(hostKDtreeArray, hostSubtreesInformations, hostDistancesArray,fictitiousNodesDepth,fictitiousNodesNumber);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return -1;
    }


    // print results on file
    writeResultsToFile(hostDistancesArray,fictitiousNodesNumber, fictitiousNodesIndexArray);

    // deallocate host ram
    hostDeallocateRam(hostKDtreeArray, hostSubtreesInformations, hostDistancesArray,datasetCoordinatesArray,fictitiousNodesIndexArray);

    /* 
     * cudaDeviceReset must be called before exiting in order for profiling and
     * tracing tools such as Nsight and Visual Profiler to show complete traces.
     */ 

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return -1;
    }

    cudaProfilerStop();

    return 0;

}


cudaError_t knnWithCuda(node *hostKDtreeArray, subtree_node *hostSubtreesInformations, float *hostDistancesArray,int fictitiousNodesDepth, int fictitiousNodesNumber)
{
    node *deviceKDtreeArray=0;
    subtree_node *deviceSubtreesInformations=0;
    float *deviceDistancesArray=0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        deviceError(deviceKDtreeArray, deviceSubtreesInformations,deviceDistancesArray);
    }

    // Allocate GPU global memory for the structures we need for implement the alghoritm 
    cudaStatus = cudaMalloc((void**)&deviceKDtreeArray,sizeof(node)*(NUMBER_OF_POINTS+fictitiousNodesNumber));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        deviceError(deviceKDtreeArray, deviceSubtreesInformations,deviceDistancesArray);
    }

    cudaStatus = cudaMalloc((void**)&deviceDistancesArray,sizeof(float)*K*(NUMBER_OF_POINTS + fictitiousNodesNumber));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        deviceError(deviceKDtreeArray, deviceSubtreesInformations,deviceDistancesArray);
    }

    cudaStatus = cudaMalloc((void**)&deviceSubtreesInformations,sizeof(subtree_node)*(fictitiousNodesNumber+1));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        deviceError(deviceKDtreeArray, deviceSubtreesInformations,deviceDistancesArray);
    }


    cudaEvent_t start, stop;
     cudaEventCreate(&start);
     cudaEventCreate(&stop);


   // Copy input structures from host memory to GPU global memory.

    cudaStatus = cudaMemcpy(deviceKDtreeArray,hostKDtreeArray,sizeof(node)*(NUMBER_OF_POINTS + fictitiousNodesNumber),cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
       deviceError(deviceKDtreeArray, deviceSubtreesInformations,deviceDistancesArray);
    }

    cudaStatus = cudaMemcpy(deviceDistancesArray,hostDistancesArray,sizeof(float)*K*(NUMBER_OF_POINTS + fictitiousNodesNumber),cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
         deviceError(deviceKDtreeArray, deviceSubtreesInformations,deviceDistancesArray);
    }


    cudaStatus = cudaMemcpy(deviceSubtreesInformations,hostSubtreesInformations,sizeof(subtree_node)*(fictitiousNodesNumber+1),cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
       deviceError(deviceKDtreeArray, deviceSubtreesInformations,deviceDistancesArray);
    }


    dim3 DimGrid(fictitiousNodesNumber+1,1);
    dim3 DimBlock(MAX_SUBTREE_DIMENSION,1,1);

    cudaEventRecord(start);

     // Launch a kernel on the GPU
    SearchFunction<<<DimGrid,DimBlock>>>(deviceKDtreeArray, deviceDistancesArray, deviceSubtreesInformations,fictitiousNodesDepth,fictitiousNodesNumber);

    cudaEventRecord(stop);

    /*
     * cudaDeviceSynchronize waits for the kernel to finish, and returns
     * any errors encountered during the launch.
     */

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
       deviceError(deviceKDtreeArray, deviceSubtreesInformations,deviceDistancesArray);
    }

    // Copy output vector from GPU global memory to host memory.
    cudaStatus = cudaMemcpy(hostDistancesArray,deviceDistancesArray, sizeof(int)*K*(NUMBER_OF_POINTS + fictitiousNodesNumber),cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
         deviceError(deviceKDtreeArray, deviceSubtreesInformations,deviceDistancesArray);
    }

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time kernel execution : %f ms\n", milliseconds);


    // deallocate GPU memory
    cudaFree(deviceDistancesArray);
    cudaFree(deviceKDtreeArray);
    cudaFree(deviceSubtreesInformations);

    return cudaStatus;

}

void deviceError(node* deviceKDtreeArray, subtree_node *deviceSubtreesInformations, float * deviceDistancesArray)
{
    cudaFree(deviceDistancesArray);
    cudaFree(deviceKDtreeArray);
    cudaFree(deviceSubtreesInformations);
}