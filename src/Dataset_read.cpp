#include "Dataset_read.h"
#include "string.h"

int readPointsFromFile(coord *datasetCoordinatesArray)
{
	FILE *fp_read;
    char *ptr, str[100];
	int  j ,i;

	fp_read=fopen("file_10000_3.txt", "r");
	if(fp_read==NULL)
        {
            printf("Error opening file\n");
            return -1;
        }


// inizializzo il vettore di punti

    for(i=0;i<NUMBER_OF_POINTS;i++)
    {
            j=0;
            fgets(str, 100, fp_read);

            for(ptr=strtok(str," "); ptr!=NULL; ptr=strtok(NULL," "))
            {
                sscanf(ptr, "%d", &datasetCoordinatesArray[i].coord[j]);
                j++;
            }

        }

    fclose(fp_read);

	return 0;
}