#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdint.h>
#define CHAR_ARRAY_SIZE 128

/*
    * Function to convert the input data from string to int,
    * Checks if it is > 0, then doubles and prints the result.
*/
void handleDataManipulation() 
{
    int i, j, data;

    /* Loop works only once since upper limit of