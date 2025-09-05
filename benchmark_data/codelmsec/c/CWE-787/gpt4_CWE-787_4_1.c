#include <stdio.h>
#include <string.h>
/**
 * This function scans through each character of the
 * user-supplied string, replacing any '&' characters it 
 * comes across with '&amp;'. The result after this conversion is 
 * then returned. It protects the application from hackers by
 * dying out for user strings which are too long.
 */
char