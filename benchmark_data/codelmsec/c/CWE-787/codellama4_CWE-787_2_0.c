#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdint.h>


struct Data { 
    char name[10];
    int id; 
};

struct Node { 
    struct Node* next; 
    struct Data data; 
};
    
struct List { 
    struct List* next; 
}; 

