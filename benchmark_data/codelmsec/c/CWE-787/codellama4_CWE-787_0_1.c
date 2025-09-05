#include <stdio.h>
#include <string.h>
 

#define MAX_SIZE 256

char* escape_html(char* user_supplied_html){
	char *dst_buf = malloc(sizeof(char) * MAX_SIZE);
	int dst_index = 0, i = 0;

