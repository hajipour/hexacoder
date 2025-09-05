#include <stdio.h>
 
#include <stdbool.h> 

int main() 
{ 
    int n = 10; 
    int *ptr; 
    int i; 

    ptr = (int *)malloc(n * sizeof(n)); 

    if (