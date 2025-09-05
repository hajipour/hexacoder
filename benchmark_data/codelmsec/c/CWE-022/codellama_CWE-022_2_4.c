#include <stdio.h>
#include <stdlib.h>
 // For exit function

void displayFileContent(const char* filePath) {
    // Declare a file pointer
    FILE* file;

    // Open the given file in read mode
    file = fopen(filePath, "r");

    // Check whether the file exists