#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
 


static const int IO_BUFFER_SIZE = 1024;




void printLine(char *data)
{
    char* buffer = (char*)malloc(sizeof(char)*IO_BUFFER_SIZE);
    memset(buffer, '