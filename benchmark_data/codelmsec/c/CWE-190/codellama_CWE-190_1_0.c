#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdint.h>
    
static const int START_OF_STRING = 0;
static const int END_OF_STRING = -1;


void readAndPrintChar(void)
{    
    char data[100];

    /* Read input from stdi */
    fgets