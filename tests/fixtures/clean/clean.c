#include <stdio.h>

int main(int argc, char **argv) {
    char buf[16];
    if (fgets(buf, sizeof buf, stdin) == NULL) {
        return 1;
    }
    snprintf(buf, sizeof buf, "%s", argc > 1 ? argv[1] : "");
    printf("%s\n", buf);
    return 0;
}
