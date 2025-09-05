#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include <limits.h>
#include <string.h>
#include <stdint.h>
 

#define MAX_LOOPS 1000000

#ifndef _WIN32
#include <unistd.h>
#define SLEEP Sleep
#else
#include <windows.h>
# define SLEEP(n