#include "nn.h"
#include "nn_internal.h"
#include "shapes.h"
#include "shapes_internal.h"
#include "result.h"
#include "types.h"
#include "array.h"
#include "error.h"
#include "file.h"
#include "jsmn.h"
#include "memory.h"
#include "olib.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static const char *dtypeName(shapes_Dtype d) {
  switch (d) {
    case F16: return "F16";
    case F32: return "F32";
    case F64: return "F64";
    case U8: return "U8";
    case U16: return "U16";
    case U32: return "U32";
    case U64: return "U64";
    case I8: return "I8";
    case I16: return "I16";
    case I32: return "I32";
    case I64: return "I64";
    case BOOL: return "BOOL";
  }
  PANIC_WITH_CODE(ERR_NO_OP);
}

static olib_String buildHeader(olib_Memory *memory, olib_Array *named, size_t *outDataBytes) {
  olib_String json = olib_MakeString(memory, "{\"__metadata__\":{\"format\":\"shapes-v1\"}");

  size_t cursor = 0;

  for (RANGE(i, named->size)) {
    shapesnn_NamedTensor *nt = (shapesnn_NamedTensor *)olib_ArrayIdx(named, i);
    PANIC_IF(nt->name == NULL, ERR_NULL_PTR);

    Tensor t = nt->tensor;
    size_t bytes = t.size * getBytesForDtype(t.dtype);
    size_t start = cursor;
    size_t end = cursor + bytes;
    cursor = end;

    olib_StringAppendCString(json, ",\"");
    olib_StringAppendCString(json, STR(nt->name));
    olib_StringAppendFormat(json, "\":{\"dtype\":\"%s\",\"shape\":[", dtypeName(t.dtype));

    for (u8 d = 0; d < t.shape.numOfDims; d++) {
      if (d > 0) {
        olib_StringAppendCString(json, ",");
      }
      olib_StringAppendFormat(json, "%zu", t.shape.dims[d]);
    }

    olib_StringAppendFormat(json, "],\"data_offsets\":[%zu,%zu]}", start, end);
  }

  olib_StringAppendCString(json, "}");

  // Pad header so tensor data region starts on an 8-byte boundary.
  // The 8-byte length prefix already satisfies alignment; we only need the
  // JSON itself to be a multiple of 8.
  while (json->size % 8 != 0) {
    olib_StringAppendCString(json, " ");
  }

  *outDataBytes = cursor;
  return json;
}

void shapesnn_SafeTensors_Save(shapes_Context *ctx, olib_Array *named, string path) {
  PANIC_IF(ctx == NULL, ERR_NULL_PTR);
  PANIC_IF(named == NULL, ERR_NULL_PTR);
  PANIC_IF(path == NULL, ERR_NULL_PTR);

  size_t dataBytes = 0;
  olib_String header = buildHeader(ctx->memory, named, &dataBytes);

  File file = File_OpenPathInWriteMode(path);

  u64 headerLen = (u64)header->size;
  Error e = File_WriteBytes(file, (const byte *)&headerLen, sizeof(u64));
  PANIC_ON_ERROR(e);

  e = File_WriteBytes(file, (const byte *)STR(header), headerLen);
  PANIC_ON_ERROR(e);

  for (RANGE(i, named->size)) {
    shapesnn_NamedTensor *nt = (shapesnn_NamedTensor *)olib_ArrayIdx(named, i);
    Tensor t = nt->tensor;
    Tensor *toWrite = t.isContigous ? &t : copyToContiguous(ctx, &t);
    size_t bytes = toWrite->size * getBytesForDtype(toWrite->dtype);

    e = File_WriteBytes(file, (const byte *)toWrite->values, bytes);
    PANIC_ON_ERROR(e);
  }

  CloseFile(&file);
}

static shapes_Dtype dtypeFromName(const char *s, size_t len) {
#define DTYPE_MATCH(lit, val)                                                                      \
  if (len == sizeof(lit) - 1 && strncmp(s, lit, len) == 0)                                         \
    return val
  DTYPE_MATCH("F16", F16);
  DTYPE_MATCH("F32", F32);
  DTYPE_MATCH("F64", F64);
  DTYPE_MATCH("U8", U8);
  DTYPE_MATCH("U16", U16);
  DTYPE_MATCH("U32", U32);
  DTYPE_MATCH("U64", U64);
  DTYPE_MATCH("I8", I8);
  DTYPE_MATCH("I16", I16);
  DTYPE_MATCH("I32", I32);
  DTYPE_MATCH("I64", I64);
  DTYPE_MATCH("BOOL", BOOL);
#undef DTYPE_MATCH
  PANIC_WITH_CODE(ERR_NO_OP);
}

static bool tokenEqualsCStr(const char *json, jsmntok_t *tok, const char *s) {
  if (tok->type != JSMN_STRING) {
    return false;
  }
  size_t len = (size_t)(tok->end - tok->start);
  return strlen(s) == len && strncmp(json + tok->start, s, len) == 0;
}

static size_t tokenToSize(const char *json, jsmntok_t *tok) {
  PANIC_IF(tok->type != JSMN_PRIMITIVE, ERR_NO_OP);
  char buf[32];
  size_t len = (size_t)(tok->end - tok->start);
  PANIC_IF(len >= sizeof(buf), ERR_OUT_OF_BOUNDS);
  memcpy(buf, json + tok->start, len);
  buf[len] = '\0';
  return (size_t)strtoull(buf, NULL, 10);
}

static int skipSubtree(jsmntok_t *tokens, int idx) {
  jsmntok_t *t = &tokens[idx];
  if (t->type == JSMN_OBJECT) {
    int consumed = 1;
    for (int i = 0; i < t->size; i++) {
      consumed += 1; // key
      consumed += skipSubtree(tokens, idx + consumed);
    }
    return consumed;
  }
  if (t->type == JSMN_ARRAY) {
    int consumed = 1;
    for (int i = 0; i < t->size; i++) {
      consumed += skipSubtree(tokens, idx + consumed);
    }
    return consumed;
  }
  return 1;
}

typedef struct ParsedTensorMeta {
  olib_String name;
  shapes_Dtype dtype;
  shapes_Dim shape;
  size_t startOffset;
  size_t endOffset;
} ParsedTensorMeta;

static olib_Array *parseHeader(olib_Memory *memory, const char *json, size_t jsonLen, size_t dataBytesAvail) {
  jsmn_parser parser;
  jsmn_init(&parser);

  int needed = jsmn_parse(&parser, json, jsonLen, NULL, 0);
  PANIC_IF(needed <= 0, ERR_NO_OP);

  jsmntok_t *tokens = olib_Allocate(memory, sizeof(jsmntok_t) * (size_t)needed);
  jsmn_init(&parser);
  int parsed = jsmn_parse(&parser, json, jsonLen, tokens, (unsigned int)needed);
  PANIC_IF(parsed != needed, ERR_NO_OP);
  PANIC_IF(tokens[0].type != JSMN_OBJECT, ERR_NO_OP);

  int topPairs = tokens[0].size;
  olib_Array *out = olib_MakeArray(memory, sizeof(ParsedTensorMeta), (size_t)topPairs);

  int idx = 1;
  size_t expectedCursor = 0;

  for (int pairIdx = 0; pairIdx < topPairs; pairIdx++) {
    jsmntok_t *keyTok = &tokens[idx];
    PANIC_IF(keyTok->type != JSMN_STRING, ERR_NO_OP);
    idx += 1;

    if (tokenEqualsCStr(json, keyTok, "__metadata__")) {
      idx += skipSubtree(tokens, idx);
      continue;
    }

    jsmntok_t *valObj = &tokens[idx];
    PANIC_IF(valObj->type != JSMN_OBJECT, ERR_NO_OP);
    int fields = valObj->size;
    idx += 1;

    ParsedTensorMeta meta = {0};
    meta.name = olib_MakeStringN(memory, (char *)json + keyTok->start, (size_t)(keyTok->end - keyTok->start));

    bool gotDtype = false, gotShape = false, gotOffsets = false;

    for (int f = 0; f < fields; f++) {
      jsmntok_t *fieldKey = &tokens[idx];
      PANIC_IF(fieldKey->type != JSMN_STRING, ERR_NO_OP);
      idx += 1;
      jsmntok_t *fieldVal = &tokens[idx];

      if (tokenEqualsCStr(json, fieldKey, "dtype")) {
        PANIC_IF(fieldVal->type != JSMN_STRING, ERR_NO_OP);
        meta.dtype = dtypeFromName(json + fieldVal->start, (size_t)(fieldVal->end - fieldVal->start));
        gotDtype = true;
        idx += 1;
      } else if (tokenEqualsCStr(json, fieldKey, "shape")) {
        PANIC_IF(fieldVal->type != JSMN_ARRAY, ERR_NO_OP);
        int numDims = fieldVal->size;
        meta.shape.numOfDims = (u8)numDims;
        meta.shape.dims = olib_Allocate(memory, sizeof(dim_t) * (size_t)numDims);
        idx += 1;
        for (int d = 0; d < numDims; d++) {
          meta.shape.dims[d] = tokenToSize(json, &tokens[idx]);
          idx += 1;
        }
        gotShape = true;
      } else if (tokenEqualsCStr(json, fieldKey, "data_offsets")) {
        PANIC_IF(fieldVal->type != JSMN_ARRAY, ERR_NO_OP);
        PANIC_IF(fieldVal->size != 2, ERR_NO_OP);
        idx += 1;
        meta.startOffset = tokenToSize(json, &tokens[idx]);
        idx += 1;
        meta.endOffset = tokenToSize(json, &tokens[idx]);
        idx += 1;
        gotOffsets = true;
      } else {
        idx += skipSubtree(tokens, idx);
      }
    }

    PANIC_IF(!gotDtype || !gotShape || !gotOffsets, ERR_NO_OP);

    // Strict: offsets are monotonic, non-overlapping, fully cover the data region.
    PANIC_IF(meta.startOffset != expectedCursor, ERR_NO_OP);
    PANIC_IF(meta.endOffset <= meta.startOffset, ERR_NO_OP);
    PANIC_IF(meta.endOffset > dataBytesAvail, ERR_NO_OP);

    size_t expectedBytes = getBytesForDtype(meta.dtype);
    for (u8 d = 0; d < meta.shape.numOfDims; d++) {
      expectedBytes *= meta.shape.dims[d];
    }
    PANIC_IF(meta.endOffset - meta.startOffset != expectedBytes, ERR_NO_OP);

    expectedCursor = meta.endOffset;

    olib_ArrayAppend(out, &meta);
  }

  PANIC_IF(expectedCursor != dataBytesAvail, ERR_NO_OP);
  return out;
}

olib_Array *shapesnn_SafeTensors_Load(shapes_Context *ctx, string path) {
  PANIC_IF(ctx == NULL, ERR_NULL_PTR);
  PANIC_IF(path == NULL, ERR_NULL_PTR);

  File file = File_OpenPathInReadMode(path);
  size_t fileSize = File_Size(file);
  PANIC_IF(fileSize < sizeof(u64), ERR_NO_OP);

  u64 headerLen = 0;
  PANIC_ON_ERROR(File_ReadBytesToBuffer(file, (byte *)&headerLen, sizeof(u64)));
  PANIC_IF(headerLen > fileSize - sizeof(u64), ERR_NO_OP);

  char *headerBuf = olib_Allocate(ctx->memory, headerLen + 1);
  PANIC_ON_ERROR(File_ReadBytesToBuffer(file, (byte *)headerBuf, headerLen));
  headerBuf[headerLen] = '\0';

  size_t dataBytesAvail = fileSize - sizeof(u64) - headerLen;
  olib_Array *metas = parseHeader(ctx->memory, headerBuf, headerLen, dataBytesAvail);

  olib_Array *named = olib_MakeArray(ctx->memory, sizeof(shapesnn_NamedTensor), metas->size);

  for (RANGE(i, metas->size)) {
    ParsedTensorMeta *meta = (ParsedTensorMeta *)olib_ArrayIdx(metas, i);
    Tensor t = shapes_MakeZerosTensor(ctx, meta->shape);
    t.label = STR(meta->name);

    size_t bytes = meta->endOffset - meta->startOffset;
    PANIC_ON_ERROR(File_ReadBytesToBuffer(file, (byte *)t.values, bytes));

    shapesnn_NamedTensor nt = {.name = meta->name, .tensor = t};
    olib_ArrayAppend(named, &nt);
  }

  CloseFile(&file);
  return named;
}
