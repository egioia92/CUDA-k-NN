#include "KDtree.h"


void hostDeallocateRam(node* hostKDtreeArray, subtree_node *hostSubtreesInformations, float * hostDistancesArray,coord *datasetCoordinatesArray,int *fictitiousNodesIndexArray)
{
	free(hostDistancesArray);
    free(hostSubtreesInformations);
    free(hostKDtreeArray);
    free(fictitiousNodesIndexArray);
}

/*
 * this function builds a balanced KDtree
 */

int buildKDtree(node *hostKDtreeArray, coord *datasetCoordinatesArray , int* index, int currentIndex ,int depth, int start, int stop, int fictitiousNodesDepth, int* fictitiousNodesIndexArray, int* fictitiousCount)
{

if(start>stop)
    return -1;                   //ho trovato una foglia

    *index=*index+1;             //index deve partire da -1
    int splitAxis= depth % N;
    int median, i;

    hostKDtreeArray[*index].name=*index; // pu� essere tolto 
    hostKDtreeArray[*index].dad=currentIndex;
    hostKDtreeArray[*index].depth=depth;

    currentIndex=*index;

    // Iteration 0: reordering the entire dataset according to x coordindates. And taking the median point as root node
    // Iteration 1: reordering the left half of the dateset according to y coordinates and taking the median point as new node
    partition(datasetCoordinatesArray,start,stop,splitAxis);
    median=(start+stop)/2;

    // Updating the spitter leaf coordinates with the median point
    for(i=0;i<N;i++)
    hostKDtreeArray[*index].coord[i]=datasetCoordinatesArray[median].coord[i];

    if(depth<=fictitiousNodesDepth)
    {
		fictitiousNodesIndexArray[*fictitiousCount]=currentIndex;
		*fictitiousCount=*fictitiousCount+1;
		hostKDtreeArray[currentIndex].left_child=buildKDtree(hostKDtreeArray,datasetCoordinatesArray, index, currentIndex, depth+1, start, median, fictitiousNodesDepth, fictitiousNodesIndexArray, fictitiousCount);
	}
	else
    hostKDtreeArray[currentIndex].left_child=buildKDtree(hostKDtreeArray,datasetCoordinatesArray, index, currentIndex, depth+1, start, median-1, fictitiousNodesDepth, fictitiousNodesIndexArray, fictitiousCount);

    hostKDtreeArray[currentIndex].right_child=buildKDtree(hostKDtreeArray,datasetCoordinatesArray, index, currentIndex, depth+1,median+1, stop, fictitiousNodesDepth, fictitiousNodesIndexArray, fictitiousCount);

    return currentIndex;

}

void partition(coord *arr,int low,int high,int splitAxis){

    int mid;

    if(low<high){
         mid=(low+high)/2;
         partition(arr,low,mid,splitAxis);
         partition(arr,mid+1,high,splitAxis);
         mergeSort(arr,low,mid,high,splitAxis);
    }
}

void mergeSort(coord* arr,int low,int mid,int high,int splitAxis){

    int i,m,k,l;
    coord temp[NUMBER_OF_POINTS];

    l=low;
    i=low;
    m=mid+1;

    while((l<=mid)&&(m<=high)){

         if(arr[l].coord[splitAxis]<=arr[m].coord[splitAxis]){
             temp[i]=arr[l];
             l++;
         }
         else{
             temp[i]=arr[m];
             m++;
         }
         i++;
    }

    if(l>mid){
         for(k=m;k<=high;k++){
             temp[i]=arr[k];
             i++;
         }
    }
    else{
         for(k=l;k<=mid;k++){
             temp[i]=arr[k];
             i++;
         }
    }

    for(k=low;k<=high;k++){
         arr[k]=temp[k];
    }
}


/*
 * this function finds and stores the start node index and the number of elements of each subtrees
 */
void findSubtrees (int fictitiousNodesDepth,int fictitiousNodesNumber, node* hostKDtreeArray,int index ,int *count, subtree_node* subtree_array)
{
	if(fictitiousNodesDepth==-1)
	{
		subtree_array[*count].startNode=0;
		subtree_array[*count].subtreeElements=NUMBER_OF_POINTS;
		return;
	}

    else if(hostKDtreeArray[index].depth+1 > fictitiousNodesDepth)
        {
            subtree_array[*count].subtreeElements= hostKDtreeArray[index].right_child - hostKDtreeArray[index].left_child;
            subtree_array[*count].startNode= hostKDtreeArray[index].left_child;
			*count=*count+1;

            if(*count==fictitiousNodesNumber)
            {   
                    subtree_array[*count].subtreeElements= NUMBER_OF_POINTS -hostKDtreeArray[index].right_child + fictitiousNodesNumber;
                    subtree_array[*count].startNode= hostKDtreeArray[index].right_child;

            } 
            else
            {	

				int parent=index;
				while(hostKDtreeArray[parent].right_child<=hostKDtreeArray[index].right_child && parent!=0)
					parent=hostKDtreeArray[parent].dad;

                subtree_array[*count].subtreeElements= hostKDtreeArray[parent].right_child - hostKDtreeArray[index].right_child;
                subtree_array[*count].startNode= hostKDtreeArray[index].right_child;
                *count=*count+1;
            }

			return;
        }

    else
{
    if(hostKDtreeArray[index].left_child!=-1)
    findSubtrees(fictitiousNodesDepth,fictitiousNodesNumber, hostKDtreeArray ,hostKDtreeArray[index].left_child, count, subtree_array);

    if(hostKDtreeArray[index].right_child!=-1)
    findSubtrees(fictitiousNodesDepth,fictitiousNodesNumber, hostKDtreeArray ,hostKDtreeArray[index].right_child, count, subtree_array);
}

 return;
}
