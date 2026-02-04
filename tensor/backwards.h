#ifndef shapes_tensor_backwards 
#define shapes_tensor_backwards

#include "common.h"
#include "result/result.h"
#include "tensor/tensor.h"

Result constructBinopBackwardpass(Context *ctx, OpType type, Tensor *a, Tensor *b, Tensor *result);


#endif
