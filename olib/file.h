#ifndef utils_lib_file_h
#define utils_lib_file_h

#include "array.h"
#include "error.h"
#include "memory.h"
#include <stdio.h>

typedef struct File {
  FILE* fd;
  string path;
} File;

File File_OpenPathInReadMode(string path);
File File_OpenPathInWriteMode(string path);
Error File_ReadBytesToBuffer(File file, byte *restrict buf, size_t numBytesToRead);
Error File_WriteBytes(File file, const byte *buf, size_t numBytesToWrite);
size_t File_Size(File file);
Array* File_ReadLines(Memory *memory, File file);

void CloseFile(File* file);

#endif
