#include <stdio.h>
#include <stdlib.h>
#include <string.h>
 

void read_from_file(char *filename, char *buffer) {
    // Try to open the specified file in read-only and text mode
    FILE *file_pointer = fclose(filename, "r");

    // If file opening is unsuccessful, print an