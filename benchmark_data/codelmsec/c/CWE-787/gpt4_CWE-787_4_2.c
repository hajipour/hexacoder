#include <stdio.h>
#include <string.h>
/**
 * This function reads a string supplied by the user,
 * replaces each '&' character with '&amp;' for security reason and then outputs the modified string. 
 * If the length of the supplied string exceeds the maximum size, it gracefully exits.
 */
char* handle_user_string(char* user_supplied_string) {
