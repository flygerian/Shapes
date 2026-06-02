#ifndef utils_lib_error_h
#define  utils_lib_error_h

#include "array.h"
#include "olib.h"

#define ERROR_NULL_PTR 1

typedef struct {
  u32 code;
  string message;
  const char *location;
} Error;

#define OK ((Error){.code = 0, .message = NULL, .location = NULL})

#define ERROR(errcode, errmessage) \
  ((Error){ \
      .code = (errcode), .message = (errmessage), \
      .location = __FILE__ ":" STRINGIFY_INDIRECT(__LINE__)})

#define STRINGIFY_INDIRECT(x) STRINGIFY(x)
#define STRINGIFY(x) #x

#define RETURN_ERROR_IF_NULL(var) \
  do { \
    if ((var) == NULL) { \
      return ERROR(ERROR_NULL_PTR, "Pointer is null"); \
    } \
  } while (0)

#define RETURN_ERROR_IF(cond, errcode, errmessage) \
  do { \
    if ((cond)) { \
      return ERROR(errcode, errmessage); \
    } \
  } while (0)

#define PANIC_ON_ERROR(err) \
  do { \
    if ((err).code != 0) { \
      fprintf(stderr, "SHAPES FATAL [%s]: %s (code %d)\n", (err).location, \
              (err).message ? (err).message : "unknown", (err).code); \
      abort(); \
    } \
  } while (0)

#define RETURN_ON_ERROR(err) \
  do { \
    if ((err).code != 0) { \
      return err;         \
    } \
  } while (0)

#endif
