#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void splitFilePath(char* file_name){
    FILE* file_pointer;
    char line_buffer[100];

  // Open the file if exists
  if ((file_pointer = fopen(file_name, "r")) == NULL)