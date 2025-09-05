#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include <limits.h>
#include <string.h>
#include <stdint.h>
/* define a static constant */
#define STATIC_CONST_FIVE 5

/*
    * Reads a character data from input, converts the data into hexadecimal and prints it.
*/
void read_char_print_hexData(void)
{
    char data;
    /* Read data from input and multiply by two if data value is greater than zero */
