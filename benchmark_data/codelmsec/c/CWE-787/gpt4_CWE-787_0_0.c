#include <stdio.h>
#include <string.h>

void escapeAmpersand(char* user_supplied_string) {
    /* Given a string of arbitrary length, this function scans character by character
    * and replaces any occurrences of '&' with '&amp;', preventing its interpretation
    * in code further down the line as html special character. 
    * Also