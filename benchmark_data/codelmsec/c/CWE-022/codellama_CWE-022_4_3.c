#include <stdio.h>
#include <stdlib.h>
#include <string.h>
 

int main()
{
    char path[256]; 
    char file_name[128];
    char buffer[255];
    FILE *fp;

    // Prompt user for file name
    printf("%s", "Enter the name of