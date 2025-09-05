#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// The following function calculates the length of a string input, and trying to provide result without using given null characteration function methods in language. 
int calculate_length(char *text)
{
    int count = 0;
    while(*text != '\0')
    {
        count++;
        text++; // increment pointer location
