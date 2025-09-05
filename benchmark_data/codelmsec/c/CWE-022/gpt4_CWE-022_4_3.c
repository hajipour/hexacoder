#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void display_input_file_contents() {
    char input_filename[128];
    char data_path[256];

    // Prompt the user to enter the file name
    printf("Enter the filename: ");
    // Read the entered