#ifndef shapes_loss_cross_entropy_h
#define shapes_loss_cross_entropy_h

#include "../result/result.h"
#include "shapes_internal.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

Result runCudaCrossEntropyForward(Context *ctx, Dtype dtype, const void *yGround, const void *logits, tensor_size_t rows, dim_t classCount, void *probs, void *loss);
Result runCudaCrossEntropyBackward(Context *ctx, Dtype dtype, const void *yGround, const void *probs, const void *gradOut, tensor_size_t rows, dim_t classCount, bool scalarGradOut, void *dLogits);

#ifdef __cplusplus
}
#endif

#endif
