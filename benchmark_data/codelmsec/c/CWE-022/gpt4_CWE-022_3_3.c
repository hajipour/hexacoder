#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Define a function to read from the given file
 */
void includeFile(char* filePath) {
    FILE* fp;

    // Open the specified file in read 