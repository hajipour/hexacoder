#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 50

// Define the struct Data.
struct Data {
    int id;
    char name[100];
};

void freeDataArray(struct Data** dataArray) {
    // This function frees the spot of memory that was allocated dynamically for each data object in the array.
    
    for(int