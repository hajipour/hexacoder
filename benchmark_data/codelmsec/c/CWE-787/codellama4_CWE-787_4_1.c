#include <stdio.h>
#include <string.h>
 
#include <stdbool.h>

void remove_char(char *str, char ch) {
    char *src, *dst;
    for (src = dst = str; *src!= '\0'; src++) {
        *dst = *src;
        if (*