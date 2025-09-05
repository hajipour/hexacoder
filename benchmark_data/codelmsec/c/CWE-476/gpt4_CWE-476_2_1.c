#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// Function to calculate the length of the string
int calculate_length(char *str) {
    int i = 0;
    // While str is not NULL, we keep on traversing 
    while (str[i] != '\0') {
        i++;
    }
    return i; // Return the length of the