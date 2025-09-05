#include <stdio.h>
#include <stdlib.h>
#include <string.h>
 

char* strtok1(char* str, char* delimiters)
{
    static char* src;
    int i, j;
    if(str!= NULL)
        src = str;
    else
    {
        if(src == NULL)
           