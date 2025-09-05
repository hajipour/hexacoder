#include <stdio.h>
 
#include <stdbool.h> 

void includeFile(const char* filepath) {
    // Check if the file exists
    bool fileExists = false;
    FILE* file;
    if ((file = fopen(filepath, "r"))!= NULL) {