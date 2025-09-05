#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdint.h>
#define CHAR_ARRAY_SIZE 50

/*
 * Reads an integer from a line, and checks whether it's over 0.
 * If it's, the function multiples it by 2 and then prints the result.
 */
void readAndDoubleIfPositive()
{
    int data, i, j;

    /* Read the integer