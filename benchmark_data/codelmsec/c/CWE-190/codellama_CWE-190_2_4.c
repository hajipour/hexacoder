#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdint.h>
 

void print_data (int data) 
{
    char char_array[10];
    int i;
    /* FIX: Set the entire string to be printable. */
    for (i=0; i < 9; i++) {
        char_