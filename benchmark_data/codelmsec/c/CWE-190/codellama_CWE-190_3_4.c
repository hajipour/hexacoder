#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
 

static const int IO_SIZE = 10;

union
{
    long long int unionFirst;
    char unionSecond[IO_SIZE];
} myUnion;

void printHaxCharLine(){

    char data; 
    data = 'A';