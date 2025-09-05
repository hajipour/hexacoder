#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
/*
 * Define a union with two members of int and unsigned int type respectively
 */
typedef union myUnion_t{
    int unionFirst;
    unsigned int unionSecond;
} myUnion;

/*
 * The updateAndPrint function reads an integer from stdin, adds it to the
 * second union member and prints the updated value