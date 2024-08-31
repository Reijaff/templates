#include <cstdio>
#include <ctime>
#include <windows.h>

int main() {
    MessageBox(NULL, "Hello, World!", "Hello, World!", MB_OK);
    printf("helloworld, %d", time(0));

    return 0;
}