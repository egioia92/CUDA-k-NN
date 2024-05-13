#include "CUDA_KDtree.cuh"

    /*
     * this function performs on GPU a parallel kNN search
     */

__global__ void SearchFunction(node* deviceKDtreeArray, float* deviceDistancesArray, subtree_node* deviceSubtreesInformations,int fictitiousNodesDepth,int fictitiousNodesNumber)
{

    // allocate in GPU shared memory an array to store subtrees
	__shared__ node sharedSubtreeArray[MAX_SUBTREE_DIMENSION];


    int id=threadIdx.x+deviceSubtreesInformations[blockIdx.x].startNode;


    if(id < deviceSubtreesInformations[blockIdx.x].startNode + deviceSubtreesInformations[blockIdx.x].subtreeElements )
        {
              // copy subtrees from GPU global memory to shared memory
              sharedSubtreeArray[threadIdx.x]=deviceKDtreeArray[id]; 
              __syncthreads();

	          int depth=deviceKDtreeArray[id].depth , index=id*K , parent, last_visited, splitAxis= depth % N;
              float radius ,dist;

              /*
               * each element of the subtree stored in shared is a root of another subtree that we explore  
               * in order to compute the knn distances  
               */

              SharedDepthFirstSearch(id,id, deviceDistancesArray, sharedSubtreeArray , &index, deviceSubtreesInformations,blockIdx.x); // vedere se possibile passare thread.x invece di id

              parent=sharedSubtreeArray[threadIdx.x].dad;  // giusto perchè siamo in shared
    	      last_visited=id;

               /*
               * once explored the subtree we go up to the parent of root in order find others possible candidates of our search.
               * then we ask if it is necessary to explore the left/right child subtree of the parent:
               * -if the number of dstances compute is less than the number required we must do it
               * -if we already got this number, only if the candidate hypersphere crosses the splitting plane we mus look on the
               *  other side of the plane by examining the other subtree.  
               */

	          while(parent != sharedSubtreeArray[0].dad)  // la radice del sotto albero è sempre nella posizione zero
	            {
	                // compute distance with parent
	            	dist=distance(sharedSubtreeArray[parent-deviceSubtreesInformations[blockIdx.x].startNode].coord, sharedSubtreeArray[threadIdx.x].coord); // tolgo l'offset da parent perchè in questo caso 																	//gli indici non coincidono  

                    if(!(deviceDistancesArray[id*K+K-1]<dist)) 
		            SortDist(deviceDistancesArray, dist, id);

            		if(index!=id*K+K) // index punta al primo elemento non inizializzato del vettore distande o meglio indica il numero di distanze inserite
		            index++;

                    // ask if it is necessary to explore the left/right child subtree of the parent

                    radius=abs(sharedSubtreeArray[threadIdx.x].coord[splitAxis]-sharedSubtreeArray[parent-deviceSubtreesInformations[blockIdx.x].startNode].coord[splitAxis]);

                    if(sharedSubtreeArray[parent-deviceSubtreesInformations[blockIdx.x].startNode].right_child!=-1 && sharedSubtreeArray[parent-deviceSubtreesInformations[blockIdx.x].startNode].right_child!=last_visited)
                     {
			             if(index<id*K+K)
			             SharedDepthFirstSearch(sharedSubtreeArray[parent-deviceSubtreesInformations[blockIdx.x].startNode].right_child, id, deviceDistancesArray, sharedSubtreeArray,&index,deviceSubtreesInformations,blockIdx.x);
			       else{
			             if((radius< deviceDistancesArray[index-1]))
					     SharedDepthFirstSearch(sharedSubtreeArray[parent-deviceSubtreesInformations[blockIdx.x].startNode].right_child, id , deviceDistancesArray, sharedSubtreeArray,&index,deviceSubtreesInformations,blockIdx.x);
                     }

        		}

        if(sharedSubtreeArray[parent-deviceSubtreesInformations[blockIdx.x].startNode].left_child!=-1 && sharedSubtreeArray[parent-deviceSubtreesInformations[blockIdx.x].startNode].left_child!=last_visited)
         {
			if(index<id*K+K)
			SharedDepthFirstSearch(sharedSubtreeArray[parent-deviceSubtreesInformations[blockIdx.x].startNode].left_child, id, deviceDistancesArray, sharedSubtreeArray, &index,deviceSubtreesInformations,blockIdx.x);
            else{
			if((radius< deviceDistancesArray[index-1]))
			  SharedDepthFirstSearch( sharedSubtreeArray[parent-deviceSubtreesInformations[blockIdx.x].startNode].left_child,  id, deviceDistancesArray, sharedSubtreeArray, &index,deviceSubtreesInformations,blockIdx.x);
				}
        }

        // update new value to continue going up in the subtree
        last_visited=parent;
		parent=sharedSubtreeArray[parent-deviceSubtreesInformations[blockIdx.x].startNode].dad;
		depth--;
		splitAxis= depth % N;
	}

    __syncthreads();

	/*
	 * once finished to explore the subtree we pass to the entire KDtree in global memory and 
	 * we continue to go up asking the same conditions. When we go up to the parent now we don't compute 
     * any distance with it becouse it is a fictitious node not present in the dataset
     */

	while(parent!=-1)
	{

    // we don't compute distance with parent  becouse it is a fictitious node not present in the dataset but we ask only for the conditions

    radius=abs(deviceKDtreeArray[id].coord[splitAxis]-deviceKDtreeArray[parent].coord[splitAxis]);

        if(deviceKDtreeArray[parent].right_child!=-1 && deviceKDtreeArray[parent].right_child!=last_visited) 
        {
			if(index<id*K+K)
				DepthFirstSearch( deviceKDtreeArray[parent].right_child, id , deviceDistancesArray, deviceKDtreeArray, &index,fictitiousNodesDepth,fictitiousNodesNumber);
			else{
					if((radius< deviceDistancesArray[index-1]))
						DepthFirstSearch(deviceKDtreeArray[parent].right_child, id, deviceDistancesArray, deviceKDtreeArray,&index,fictitiousNodesDepth,fictitiousNodesNumber);
            }

		}

        if(deviceKDtreeArray[parent].left_child!=-1 && deviceKDtreeArray[parent].left_child!=last_visited) 
        {
			if(index<id*K+K)
				DepthFirstSearch( deviceKDtreeArray[parent].left_child, id, deviceDistancesArray, deviceKDtreeArray, &index,fictitiousNodesDepth,fictitiousNodesNumber);
            else{
				if((radius< deviceDistancesArray[index-1]))
			  		DepthFirstSearch( deviceKDtreeArray[parent].left_child , id , deviceDistancesArray, deviceKDtreeArray, &index,fictitiousNodesDepth,fictitiousNodesNumber);
				}

        }

    //  update new value to continue going up in the Kdtree
            last_visited=parent;
			parent=deviceKDtreeArray[parent].dad;
			depth--;
			splitAxis= depth % N;
	}

__syncthreads();

  }

  }




/*
 * this finction sorts the compute distance in increasing order 
 */   

__device__ void SortDist(float *deviceDistancesArray, float dist, int id)
{
        int i=id*K, index;
        while((dist>deviceDistancesArray[i])&&(i<K+id*K))
        i++;

	index=i;

	i=id*K+K-1;	

        while(i>index)
        {

            deviceDistancesArray[i]=deviceDistancesArray[i-1];
            i--;
        }

	deviceDistancesArray[index]=dist;
}

/*
 * this finction does a depth serch of a subtree of KDtree stored in global memory starting from the root of subtree
 * and compute the distance between the query point and all nodes of subtree
 */

__device__ void DepthFirstSearch(int root,int query, float* deviceDistancesArray, node *deviceKDtreeArray, int *index,int fictitiousNodesDepth,int fictitiousNodesNumber)
{
	int current_node=root;
	int id=query;
	float dist;

    // appena entro calcolo la distanza tra root e query se root non è un fittizio

    if(current_node!=query && deviceKDtreeArray[current_node].depth>fictitiousNodesDepth) // la prima condizione non serve
    {
            dist=distance(deviceKDtreeArray[current_node].coord, deviceKDtreeArray[query].coord);

            if(!(deviceDistancesArray[id*K+K-1]<dist))
			SortDist(deviceDistancesArray, dist, id);

            if(*index!=id*K+K)
			*index=*index+1;
    }

    current_node++;

	while(deviceKDtreeArray[current_node].depth>deviceKDtreeArray[root].depth && current_node<NUMBER_OF_POINTS+fictitiousNodesNumber)
	{

		if(current_node!=query && deviceKDtreeArray[current_node].depth>fictitiousNodesDepth) // la seconda condizione è essenziale perchè non calcolo la distanza con i fittizi
		{
			dist=distance(deviceKDtreeArray[current_node].coord, deviceKDtreeArray[query].coord);

            if(!(deviceDistancesArray[id*K+K-1]<dist))
			SortDist(deviceDistancesArray, dist, id);

            if(*index!=id*K+K)
			*index=*index+1;

		}

    current_node++;

	}


}

/*
 * this finction does a depth serch stored in shared memory starting from the root of subtree
 * and computes the distance between the query point and all nodes of subtree
 */

__device__ void SharedDepthFirstSearch(int root,int query, float* deviceDistancesArray, node *sharedSubtreeArray, int *index, subtree_node* deviceSubtreesInformations, int blockIdx)
{
     int current_node=root-deviceSubtreesInformations[blockIdx].startNode+1;
	int id=query;
	float dist;

	// appena entro calcolo la distanza con il nodo se è diverso da se stesso
	if(root!=query)
	{
		dist=distance(sharedSubtreeArray[root-deviceSubtreesInformations[blockIdx].startNode].coord, sharedSubtreeArray[query-deviceSubtreesInformations[blockIdx].startNode].coord);

            if(!(deviceDistancesArray[id*K+K-1]<dist))
			SortDist(deviceDistancesArray, dist, id);

            if(*index!=id*K+K)
			*index=*index+1;

    }



	while(sharedSubtreeArray[current_node].depth>sharedSubtreeArray[root-deviceSubtreesInformations[blockIdx].startNode].depth && current_node<deviceSubtreesInformations[blockIdx].subtreeElements) // no minore uguale perchè crea casini
	{

		dist=distance(sharedSubtreeArray[current_node].coord, sharedSubtreeArray[query-deviceSubtreesInformations[blockIdx].startNode].coord);

            if(!(deviceDistancesArray[id*K+K-1]<dist))
			SortDist(deviceDistancesArray, dist, id);

            if(*index!=id*K+K)
			*index=*index+1;

    current_node++;

	}
}


/*
 * this finction  computes the distances
 */

__device__ float distance(int *a, int *b)
{
  float d, dist = 0;
  int i;
    for(i=0; i < N; i++)
    {
        d = a[i] - b[i];
        dist += d*d;
    }
  dist=(float)sqrt((double)dist);
    return dist;
}