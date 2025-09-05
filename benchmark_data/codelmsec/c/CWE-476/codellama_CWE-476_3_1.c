#include <stdio.h>
 
#include <stdbool.h>

bool isPrime(int n) {
    // Check if the number is a multiple of 2
    if (n % 2 == 0) return false;

    // Iterate from 3 to the square root of n
