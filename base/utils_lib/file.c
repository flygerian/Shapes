#include "result/result.h"
#include "utils_lib/array.h"
#include "utils_lib/utils_lib.h"
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "file.h"
#include "utils_lib/error.h"


File File_OpenPathInReadMode(string path) {
  FILE* fileHandle = fopen(path, "rb");
  PANIC_IF_NULL(fileHandle);

  return (File) {.fd = fileHandle, .path = path};
}

Error File_ReadBytesToBuffer(File file, byte *restrict buf, size_t numBytesToRead) {
  RETURN_ERROR_IF_NULL(buf);
  RETURN_ERROR_IF_NULL(file.fd);

  size_t res = fread(buf, 1, numBytesToRead, file.fd);
  RETURN_ERROR_IF(res != numBytesToRead, ERR_EOF, "End of file or incomplete read");

  return OK;
}

Array* File_ReadLines(Memory *memory, File file) { 
  PANIC_IF_NULL(file.fd);

  char line[1024];
  Array *lines = MakeDynamicArray(memory, sizeof(line));

  size_t totalLines = 0;
  while (fgets(line, sizeof(line), file.fd)) {
     line[strcspn(line, "\n")] = '\0'; 
     Array_Append(lines, line);
     totalLines += 1;
  }

  return lines;
}

void CloseFile(File* file) {
  fclose(file->fd);
}

