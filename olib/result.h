#ifndef shapes_error_h
#define shapes_error_h

#include "stdlib.h"
#include <stddef.h>
#include "stdio.h"

typedef enum {
  OK,
  ERR_NO_OP,
  ALLOCATION_FAILED,
  CUDA_OP_FAILED,
  NULL_CONTEXT,
  NO_DEVICE_ON_CONTEXT,
  ERR_DEVICE_MISMATCH,
  ALLOCATING_ZERO,
  ERR_DTYPE_MISMATCH,
  ERR_DIM_MISMATCH,
  ERR_OUT_OF_BOUNDS,
  ERR_SUM_DIM_OUT_OF_BOUNDS,
  ERR_NULL_PTR,
  ERR_INVALID_RANGE,
  ERR_NULL_TENSOR_PROVIDED,
  ERR_NULL_SHAPE_PROVIDED,
  ERR_RESHAPE_DIM_MISMATCH,
  ERR_INVALID_TRANSPOSE,
  ERR_MAX_N_DIMS_EXCEEDED,
  ERR_CANNOT_FREE_VIEW_TENSOR,
  ERR_MATMUL_MIN_2D,
  ERR_MATMUL_INNER_DIM_MISMATCH,
  ERR_TANH_VALUE_NOT_FLOAT,
  ERR_RELU_VALUE_NOT_FLOAT,
  ERR_POW_VALUE_NOT_FLOAT,
  ERR_EXP_VALUE_NOT_FLOAT,
  ERR_NEGATE_UNSUPPORTED_DTYPE,
  ERR_MEAN_VALUE_NOT_FLOAT,
  ERR_LOG_VALUE_NOT_FLOAT,
  ERR_ABS_VALUE_NOT_SIGNED,
  ERR_SQRT_VALUE_NOT_FLOAT,
  ERR_NOT_A_BINOP,
  ERR_ZERO_DIM_TENSOR_ADVANCED_INDEXING,
  ERR_ONLY_INT_TYPE_ALLOWED,
  ERR_TRUNCATING_CAST,
  ERR_SIGN_MISMATCH_CAST,
  ERR_OUT_OF_MEMORY,
  ERR_COPY_REQUIRES_INITIALIZED_TENSORS,
  ERR_COPY_REQUIRES_TENSORS_OF_THE_SAME_SIZE,
  ERR_COPY_DESTINATION_VIEW,
  ERR_COPY_SAME_DTYPE,
  ERR_STD_NOT_FLOAT_TYPE,
  ERR_STD_REQUIRES_AT_LEAST_TWO_VALUES,
  ERR_LEARNING_RATE_CANNOT_BE_ZERO_OR_NEGATIVE,
  ERR_SGD_PARAMS_HAVE_TO_BE_CONTIGOUS,
  ERR_SGD_PARAMS_HAVE_TO_BE_FLOAT,
  ERR_SGD_PARAMS_NUMBER_MISMATCH,
  ERR_SGD_PARAMS_GRAD_DTYPE_MISMATCH,
  ERR_CONV2D_INVALID_NUM_TENSOR_DIM,
  ERR_CONV2D_IN_CHANNELS_ZERO,
  ERR_CONV2D_OUT_CHANNELS_ZERO,
  ERR_CONV2D_KERNEL_NOT_2D,
  ERR_CONV2D_KERNEL_NOT_FLOAT,
  ERR_CONV2D_KERNEL_STRIDE_ZERO,
  ERR_CONV2D_KERNEL_NOT_CONTIGOUS,
  ERR_ADAM_NULL_TRIPLETS,
  ERR_ADAM_NULL_M,
  ERR_ADAM_NULL_V,
  ERR_ADAM_NULL_PARAM,
  ERR_ADAM_NULL_GRAD,
  ERR_ADAM_ONLY_FLOAT_TENSORS,
  ERR_ADAM_PARAMS_SIZE_MISMATCH,

  ERR_CONCAT_TENSOR_IS_NULL,
  ERR_CONCAT_TARGET_DIM_IS_OUT_OF_BOUNDS,
  ERR_CONCAT_TENSOR_DOES_NOT_FIT_IN_TARGET_DIM,
  ERR_CONCAT_TENSOR_NOT_CONTIGOUS,
  ERR_CONCAT_TENSOR_NOT_SAME_DTYPE,
  ERR_CONCAT_TENSORS_UNEQUAL_DIMS,
  ERR_CONCAT_SOURCE_TENSOR_CANNOT_HAVE_ZERO_DIMS,
  ERR_COPY_CTX_DEVICE_IS_NULL,
  ERR_DIFFERENT_CTX_TENSORS_PASSED,
  NON_CONTIGOUS_MOVE_TENSOR,
  NOT_A_DENSE_LAYER,
  OPTIMIZER_OP_NOT_FOUND,
  LAYER_OP_NOT_FOUND,
  BACKWARD_TENSOR_OP_NOT_FOUND,
  BATCH_NORM_ZERO_DIM_NOT_ALLOWED,
  FILE_OPEN_FAILED,
  TENSORS_CANNOT_BE_BROADCASTED,
  ZERO_LAYERS_PASSED,
  NON_LAYER_OP_PASSED,
  OP_NOT_SEQUENTIAL,
  ARRAY_ELEM_SIZE_MISMATCH,
  ERR_EXPAND_FIXED_ARRAY,
  ERR_EOF,
  ERR_STACKING_LESS_THAN_TWO_TENSORS,
  ERR_CUDA_BLOCK_MISMATCH,
  ERR_CUDA_BLOCK_NO_ALLOCATION_CHECKPOINT
} Result;

#define PANIC_IF(cond, errCode)                                                                    \
  do {                                                                                             \
    if ((cond)) {                                                                                  \
      fprintf(stderr, "SHAPES FATAL [%s:%d]: %d\n", __FILE__, __LINE__, errCode);                  \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)

#define PANIC_IF_NULL(var)                                                                         \
  do {                                                                                             \
    if ((var) == NULL) {                                                                           \
      fprintf(stderr, "SHAPES FATAL [%s:%d]: %d\n", __FILE__, __LINE__, ERR_NULL_PTR);             \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)

#define PANIC_WITH_MSG_IF(cond, msg)                                                                    \
  do {                                                                                             \
    if ((cond)) {                                                                                  \
      fprintf(stderr, "SHAPES FATAL [%s:%d]: %s\n", __FILE__, __LINE__, msg);                  \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)

#define PANIC_WITH_CODE(errCode)                                                                   \
  do {                                                                                             \
    fprintf(stderr, "SHAPES FATAL [%s:%d]: %d\n", __FILE__, __LINE__, errCode);                    \
    abort();                                                                                       \
  } while (0)

#endif
