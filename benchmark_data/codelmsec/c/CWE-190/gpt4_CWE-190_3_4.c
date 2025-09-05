#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
/* Define a type 'my_data' to hold union type */
typedef union {
    unsigned int unionFirst;
    unsigned int unionSecond;
} my_data;

/* Function to capture data from stdin, increase it by 1 and print the result */
void read_and_increase_data() {
    /* Invoke a my_data object */
