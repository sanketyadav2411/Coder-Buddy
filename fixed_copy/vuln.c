#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    char buf[16];
    gets(buf);
    strcpy(buf, argv[1]);
    system(buf);
    return 0;
}
