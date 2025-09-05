#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void locate_and_display_file(char *input_filename) {
    char data_path[512];

    // Open the input file in read mode
    FILE* fp = fopen(strcat("/home/user/data/", input_filename), "