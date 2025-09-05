#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
/*
    * Define a union to encapsulate two forms (unsigned characters and integer) of the same data.
*/
typedef union
{
    unsigned int unionFirst;
    unsigned char unionSecond[sizeof(unsigned int)];
} data_hybrid_t;

/* Multiply data function, it receives an integer, increments it by 1 and print