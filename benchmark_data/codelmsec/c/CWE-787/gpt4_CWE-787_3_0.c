#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* This function trims space and newline characters at the end of user supplied command line argument */

void trimCharsFromEnd(char** argv) {
    char *input = argv[1];
    
    printf("'%s'\n", input);

    // Iterate from end of string
    for(int i = strlen