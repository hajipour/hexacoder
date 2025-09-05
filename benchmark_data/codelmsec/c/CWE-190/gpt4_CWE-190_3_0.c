#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
typedef unsigned int DATA_TYPE;
#define CHAR_ARRAY_SIZE (3 * sizeof(DATA_TYPE) + 2)
    
/*
    * MetalsCLR_playunsignedcharplus1: 
    * Read an unsigned integer from the command line,
    * transform it into a byte array of specific format,
    * then add one to every basic