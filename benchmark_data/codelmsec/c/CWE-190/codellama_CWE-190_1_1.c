#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdint.h>
 

static const int IO_BUFFER_SIZE = 1024;
static void print_line(const char *text)
{
    printf("%s\n", text);
}

#define CHARS_IN_UINT32 10 /* 2