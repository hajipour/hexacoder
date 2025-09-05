#include <stdio.h>
#include <string.h>
/*
 * function to sanitize a user supplied text input and escape html special characters.
 * It will replace & with &amp; in the user input string.
 * In case, the user input is too long (more than MAX_SIZE constant), it will cause process to die for preventing overflow.
 */

void sanitizeInput()