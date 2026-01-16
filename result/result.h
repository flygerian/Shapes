#ifndef shapes_error_h
#define shapes_error_h

typedef enum {
  OK,
  ERR_DTYPE_MISMATCH,
  ERR_DIM_MISMATCH,
  ERR_OUT_OF_BOUNDS,
  ERR_NULL_PTR
} Result;

#endif
