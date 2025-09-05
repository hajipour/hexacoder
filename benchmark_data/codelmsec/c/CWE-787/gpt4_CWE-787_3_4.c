#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// This function takes a command line produced string, trims whitespace 
// from its end and then prints it.
int main(int argc, char *argv[]) {
   // Check if exactly one commandline parameter is provided
   if(argc!=2){
      printf("One input required \n");
      return 