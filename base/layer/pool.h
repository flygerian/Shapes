#ifndef shapes_layer_pool_h
#define shapes_layer_pool_h

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

Result runCudaMaxPool2d(Context *ctx, Dtype dtype, const void *input, dim_t batch, dim_t channels,
                        dim_t h, dim_t w, dim_t kH, dim_t kW, u8 stride, void *output);
Result runCudaMaxPool2dWithIndices(Context *ctx, Dtype dtype, const void *input, dim_t batch,
                                   dim_t channels, dim_t h, dim_t w, dim_t kH, dim_t kW,
                                   u8 stride, void *output, void *indices);
Result runCudaMaxPool2dBackward(Context *ctx, Dtype dtype, const void *input, const void *gradOut,
                                dim_t batch, dim_t channels, dim_t h, dim_t w, dim_t kH,
                                dim_t kW, u8 stride, void *dX);
Result runCudaMaxPool2dBackwardWithIndices(Context *ctx, Dtype dtype, const void *gradOut,
                                           const void *indices, tensor_size_t numGradValues,
                                           void *dX);
Result runCudaAdaptiveAvgPool2d(Context *ctx, Dtype dtype, const void *input, dim_t batch,
                                dim_t channels, dim_t h, dim_t w, dim_t outH, dim_t outW,
                                void *output);
Result runCudaAdaptiveAvgPool2dBackward(Context *ctx, Dtype dtype, const void *gradOut, dim_t batch,
                                        dim_t channels, dim_t h, dim_t w, dim_t outH, dim_t outW,
                                        void *dX);

#ifdef __cplusplus
}
#endif

#endif
