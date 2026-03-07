#ifndef shapes_layer_batch_norm_h
#define shapes_layer_batch_norm_h

#include "common.h"
#include "tensor/tensor.h"

Result BatchNormBackward(Context *ctx, Tensor *x2d, Tensor *grad2d, Tensor *gamma, f32 epsilon,
                         Tensor *dX, Tensor *dGamma, Tensor *dBeta);

#endif
