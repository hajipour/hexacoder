#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdint.h>
#define CHAR_ARRAY_SIZE 128

/*
    * Reads and checks input from stdin. It then, if the read data is greater than 
    * 0, multiplies it by two and prints the result.
*/
void readDataAndPrintResult(void)
{
    int data;
    unsigned int i, j;

   