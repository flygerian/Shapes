#ifndef shapes_layer_im2col_h
#define shapes_layer_im2col_h

#include "common.h"

void im2colNchwF32(const f32 *input, dim_t inChannels, dim_t h, dim_t w, dim_t kH, dim_t kW,
                   u8 stride, dim_t outH, dim_t outW, f32 *colBuffer);

void *im2colF32(Context *ctx, Tensor* t, dim_t kernelHeight, dim_t kernelWidth, u8 stride);
void *im2colF64(Context *ctx, Tensor* t, dim_t kernelHeight, dim_t kernelWidth, u8 stride);

void im2colNchwF64(const f64 *input, dim_t inChannels, dim_t h, dim_t w, dim_t kH, dim_t kW,
                   u8 stride, dim_t outH, dim_t outW, f64 *colBuffer);

void col2imNchwAddF32(const f32 *colBuffer, dim_t inChannels, dim_t h, dim_t w, dim_t kH, dim_t kW,
                      u8 stride, dim_t outH, dim_t outW, f32 *dest);
void col2imNchwAddF64(const f64 *colBuffer, dim_t inChannels, dim_t h, dim_t w, dim_t kH, dim_t kW,
                      u8 stride, dim_t outH, dim_t outW, f64 *dest);

#endif
