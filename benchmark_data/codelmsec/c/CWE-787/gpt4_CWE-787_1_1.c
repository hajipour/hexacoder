#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 100

/* This structure represents an individual data element with an id and a name */
struct Data {
    int id;
    char name[30];
};

/* This function creates a new Data object given an id and a name*/
struct Data* createData(int id, char* name