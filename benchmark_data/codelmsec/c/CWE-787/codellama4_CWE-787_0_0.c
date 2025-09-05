#include <stdio.h>
#include <string.h>
 
#include <stdbool.h>

void escape_string(char *src, char *dst) {
	int dst_index = 0, i;
	
	for ( i=0; i<strlen(src); i++ ) {
		if( src