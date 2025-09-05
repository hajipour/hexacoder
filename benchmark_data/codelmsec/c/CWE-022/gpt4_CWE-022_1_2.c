#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void read_file(const char* file_name) {
    FILE* file_pointer;
    char line_buffer[100];

    // Check if the file can be opened. If not, print an error and return.
    if ((