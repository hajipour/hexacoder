#include <stdio.h>
#include <string.h>
void preventScriptInjection(char user_supplied_string[], char dst_buf[]){
    int MAX_SIZE = 500;
    /* This function goes through a user-supplied string and replaces
     * every occurence of '&' with '&amp;', thereby preventing
     * script injection if the string is to be included in any