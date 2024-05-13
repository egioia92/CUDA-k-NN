#include "Results_write.h"

void writeTreeToFile(node* hostKDtreeArray, int Number)
{
    FILE* fp_write_tree;
    int i;

    fp_write_tree=fopen("file_results_tree.txt", "w");

    if(fp_write_tree==NULL)
        {
            printf("Error closing file_tree\n");
        }

    for(i=0; i<Number; i++)
    {
        fprintf(fp_write_tree, "name: %d  coord: %d  %d  padre: %d depth: %d   fig_s:  %d fig_d: %d\n\n", hostKDtreeArray[i].name, hostKDtreeArray[i].coord[0], hostKDtreeArray[i].coord[1], hostKDtreeArray[i].dad, hostKDtreeArray[i].depth, hostKDtreeArray[i].left_child, hostKDtreeArray[i].right_child);

    }

}

 void writeResultsToFile (float* hostDistancesArray,int fictitiousNodesNumber, int* fictitiousNodesIndexArray)
{
    FILE* fp_write;
    int i, j, index_false=0;

    fp_write=fopen("file_results_par.txt", "w");

    if(fp_write==NULL)
        {
            printf("Error closing file\n");
        }

    for(i=0; i<(NUMBER_OF_POINTS + fictitiousNodesNumber); i++)
    {
		if((i!=fictitiousNodesIndexArray[index_false])||(index_false>=fictitiousNodesNumber))
        {
			fprintf(fp_write, "Query:  %d\n\n", i);

			for(j=i*K; j<i*K+K; j++)
			fprintf(fp_write,"%f\n", hostDistancesArray[j]);

	        fprintf(fp_write, "\n\n");

		}

		else
			index_false++;

	}
}