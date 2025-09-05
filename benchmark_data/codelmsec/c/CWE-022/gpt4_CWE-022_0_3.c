#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX_PATH_LEN 256

// Function to open the user's requested file and 
// then print its content line by line.
void printFileContent(const char* input_path) {
    FILE *fp = fopen(input