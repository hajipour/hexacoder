#include <stdio.h>
#include <stdlib.h>
#include <string.h>
 

#define SIZ 100

int main(int argc, char **argv){
    char input[SIZ];
    int i, j, k = 0;
    
    gets(input);
    
    for(i = strlen(input)