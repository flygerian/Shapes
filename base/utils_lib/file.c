#include "result/result.h"
#include "utils_lib/array.h"
#include <stddef.h>
#include <stdio.h>
#include "file.h"
#include "utils_lib/error.h"

File File_OpenPathInReadMode(string path) {
  FILE* fileHandle = fopen(path, "rb");
  PANIC_IF_NULL(fileHandle);

  return (File) {.fd = fileHandle, .path = path};
}

Error File_ReadBytesToBuffer(File *file, void *buf, size_t numBytesToRead) {
  RETURN_ERROR_IF_NULL(file);
  RETURN_ERROR_IF_NULL(buf);
  RETURN_ERROR_IF_NULL(file->fd);

  size_t res = fread(buf, 1, numBytesToRead, file->fd);
  RETURN_ERROR_IF(res != numBytesToRead, ERR_EOF, "End of file or incomplete read");

  return OK;
}

void CloseFile(File* file) {
  fclose(file->fd);
}
