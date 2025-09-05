#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 20 

// Define a struct to hold person Data
struct Data {
   int id;
   char name[50];
};

// Function that creates an instance of Data by allocating memory and initializing it with the provided information
struct Data* createData(int id, char* name) {
