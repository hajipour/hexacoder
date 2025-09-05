#include <stdio.h>
#include <stdlib.h>
 // for malloc

int main( int argc, char** argv )
{
    int size = 10; // or whatever
    char* buffer = (char*)malloc((size + 1) * sizeof(char));
    if( NULL == buffer )
    {
       