#include "result.h"
#include "array.h"
#include "olib.h"
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "file.h"
#include "error.h"


File File_OpenPathInReadMode(string path) {
  FILE* fileHandle = fopen(path, "rb");
  PANIC_IF_NULL(fileHandle);

  return (File) {.fd = fileHandle, .path = path};
}

File File_OpenPathInWriteMode(string path) {
  FILE* fileHandle = fopen(path, "wb");
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

Error File_WriteBytes(File file, const byte *buf, size_t numBytesToWrite) {
  RETURN_ERROR_IF_NULL(buf);
  RETURN_ERROR_IF_NULL(file.fd);

  size_t res = fwrite(buf, 1, numBytesToWrite, file.fd);
  RETURN_ERROR_IF(res != numBytesToWrite, ERR_NO_OP, "Incomplete write");

  return OK;
}

size_t File_Size(File file) {
  PANIC_IF_NULL(file.fd);
  long original = ftell(file.fd);
  PANIC_IF(original < 0, ERR_NO_OP);
  PANIC_IF(fseek(file.fd, 0, SEEK_END) != 0, ERR_NO_OP);
  long size = ftell(file.fd);
  PANIC_IF(size < 0, ERR_NO_OP);
  PANIC_IF(fseek(file.fd, original, SEEK_SET) != 0, ERR_NO_OP);
  return (size_t)size;
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

