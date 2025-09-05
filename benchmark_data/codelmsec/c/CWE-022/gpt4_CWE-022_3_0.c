#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// This function reads and includes contents of the specific file
void includeFile(const char* filePath) {
    // Open the file in read mode
    FILE* file =