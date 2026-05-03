#include "utils_lib/array.h"
#include "utils_lib/error.h"
#include <stdio.h>

typedef struct File {
  FILE* fd;
  string path;
} File;

File File_OpenPathInReadMode(string path);
Error File_ReadBytesToBuffer(File file, byte *restrict buf, size_t numBytesToRead);

void CloseFile(File* file);
