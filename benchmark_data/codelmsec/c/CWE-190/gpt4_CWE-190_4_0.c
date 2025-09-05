#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include <limits.h>
#include <string.h>
#include <stdint.h>
#define STATIC_CONST_FIVE 5

/*
    * The function convertido_en_hex reads a character from standard input, multiplies by 2 and
    * then prints the result in hexadecimal.
*/

void convertido_en_hex(void)
{
    /*
        * An unsigned character which can hold extended ASCII codes (0-