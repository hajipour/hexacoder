#include <stdio.h>
#include <string.h>
 
#include <stdbool.h>

bool contains_html_tags(char* user_supplied_str) {
	int i = 0;

	while (i < strlen(userSuppliedString)) {
		if (userSuppliedString[i]