package shapes

/*
#cgo CFLAGS: -I../base
#include "result/result.h"
*/
import "C"
import "fmt"

// ResultError converts a non-OK C Result code into a Go error.
func ResultError(r uint32) error {
	return fmt.Errorf("shapes: %s", ResultString(r))
}

// ResultString returns a human-readable string for a C Result code.
func ResultString(r uint32) string {
	switch r {
	case C.OK:
		return "ok"
	case C.ERR_NO_OP:
		return "no op"
	case C.ERR_DTYPE_MISMATCH:
		return "dtype mismatch"
	case C.ERR_DIM_MISMATCH:
		return "dimension mismatch"
	case C.ERR_OUT_OF_BOUNDS:
		return "out of bounds"
	case C.ERR_SUM_DIM_OUT_OF_BOUNDS:
		return "sum dimension out of bounds"
	case C.ERR_NULL_PTR:
		return "null pointer"
	case C.ERR_INVALID_RANGE:
		return "invalid range"
	case C.ERR_NULL_TENSOR_PROVIDED:
		return "null tensor provided"
	case C.ERR_NULL_SHAPE_PROVIDED:
		return "null shape provided"
	case C.ERR_RESHAPE_DIM_MISMATCH:
		return "reshape dimension mismatch"
	case C.ERR_INVALID_TRANSPOSE:
		return "invalid transpose"
	case C.ERR_MAX_N_DIMS_EXCEEDED:
		return "max number of dimensions exceeded"
	case C.ERR_CANNOT_FREE_VIEW_TENSOR:
		return "cannot free view tensor"
	case C.ERR_MATMUL_MIN_2D:
		return "matmul requires at least 2D tensors"
	case C.ERR_MATMUL_INNER_DIM_MISMATCH:
		return "matmul inner dimension mismatch"
	case C.ERR_TANH_VALUE_NOT_FLOAT:
		return "tanh requires float values"
	case C.ERR_POW_VALUE_NOT_FLOAT:
		return "pow requires float values"
	case C.ERR_EXP_VALUE_NOT_FLOAT:
		return "exp requires float values"
	case C.ERR_NEGATE_UNSUPPORTED_DTYPE:
		return "negate unsupported dtype"
	case C.ERR_MEAN_VALUE_NOT_FLOAT:
		return "mean requires float values"
	case C.ERR_LOG_VALUE_NOT_FLOAT:
		return "log requires float values"
	case C.ERR_NOT_A_BINOP:
		return "not a binary operation"
	case C.ERR_ZERO_DIM_TENSOR_ADVANCED_INDEXING:
		return "zero-dim tensor advanced indexing not allowed"
	case C.ERR_ONLY_INT_TYPE_ALLOWED:
		return "only int tensors allowed"
	case C.ERR_TRUNCATING_CAST:
		return "truncating cast"
	case C.ERR_SIGN_MISMATCH_CAST:
		return "sign mismatch in cast"
	case C.ERR_OUT_OF_MEMORY:
		return "out of memory - arena exhausted"
	case C.ERR_STD_NOT_FLOAT_TYPE:
		return "std requires float values"
	case C.ERR_STD_REQUIRES_AT_LEAST_TWO_VALUES:
		return "std requires at least two values"
	case C.ERR_LEARNING_RATE_CANNOT_BE_ZERO_OR_NEGATIVE:
		return "learning rate cannot be zero or negative"
	case C.ERR_SGD_PARAMS_HAVE_TO_BE_CONTIGOUS:
		return "sgd params must be contiguous"
	case C.ERR_SGD_PARAMS_HAVE_TO_BE_FLOAT:
		return "sgd params must be float tensors"
	case C.ERR_SGD_PARAMS_NUMBER_MISMATCH:
		return "sgd params number mismatch"
	case C.ERR_SGD_PARAMS_GRAD_DTYPE_MISMATCH:
		return "sgd params grad dtype mismatch"
	case C.ERR_CONV2D_INVALID_NUM_TENSOR_DIM:
		return "conv2d invalid tensor dimension"
	case C.ERR_CONV2D_IN_CHANNELS_ZERO:
		return "conv2d in channels cannot be zero"
	case C.ERR_CONV2D_OUT_CHANNELS_ZERO:
		return "conv2d out channels cannot be zero"
	case C.ERR_CONV2D_KERNEL_NOT_2D:
		return "conv2d kernel must be 2D"
	case C.ERR_CONV2D_KERNEL_NOT_FLOAT:
		return "conv2d kernel must be float"
	case C.ERR_CONV2D_KERNEL_STRIDE_ZERO:
		return "conv2d kernel stride cannot be zero"
	case C.ERR_ADAM_NULL_TRIPLETS:
		return "adam triplets array is null"
	case C.ERR_ADAM_NULL_M:
		return "adam m tensor is null"
	case C.ERR_ADAM_NULL_V:
		return "adam v tensor is null"
	case C.ERR_ADAM_NULL_PARAM:
		return "adam param tensor is null"
	case C.ERR_ADAM_NULL_GRAD:
		return "adam paramGrad tensor is null"
	case C.ERR_ADAM_ONLY_FLOAT_TENSORS:
		return "adam requires float tensors"
	case C.ERR_ADAM_PARAMS_SIZE_MISMATCH:
		return "adam tensor sizes must match"
	default:
		return fmt.Sprintf("unknown error (%d)", int(r))
	}
}
