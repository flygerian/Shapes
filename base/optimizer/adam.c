#include "common.h"
#include "result/result.h"
#include "tensor/binary_op_helpers.h"
#include "tensor/tensor_internal.h"
#include <math.h>

#ifndef rsqrtf
static inline float rsqrtf(float x) {
  return 1.0f / sqrtf(x);
}
#endif
Result Adam(Context *ctx, AdamData *triplets, size_t numParameters, f32 b1, f32 b2, size_t step,
            f32 a, f32 epsilon) {
  if (triplets == NULL) {
    return ERR_ADAM_NULL_TRIPLETS;
  }

  for (size_t tripletIdx = 0; tripletIdx < numParameters; tripletIdx++) {
    AdamData trip = triplets[tripletIdx];

    if (trip.m == NULL) {
      return ERR_ADAM_NULL_M;
    }

    if (trip.v == NULL) {
      return ERR_ADAM_NULL_V;
    }

    if (trip.param == NULL) {
      return ERR_ADAM_NULL_PARAM;
    }

    if (trip.param->grad == NULL) {
      return ERR_ADAM_NULL_GRAD;
    }

    if (isNotFloatType(trip.m) || isNotFloatType(trip.param) || isNotFloatType(trip.v) ||
        isNotFloatType(trip.param->grad)) {
      return ERR_ADAM_ONLY_FLOAT_TENSORS;
    }

    if (trip.param->size != trip.m->size || trip.param->size != trip.v->size) {
      return ERR_ADAM_PARAMS_SIZE_MISMATCH;
    }

    if (!trip.param->isContigous || !trip.param->grad->isContigous || !trip.m->isContigous ||
        !trip.v->isContigous) {
      return ERR_ADAM_PARAMS_MUST_BE_CONTIGUOUS;
    }

    f32 *gradVals = trip.param->grad->values;
    f32 *pVals = trip.param->values;

    f32 *mVals = trip.m->values;
    f32 *vVals = trip.v->values;

    f32 oneMinusB1 = 1.0f - b1;
    f32 oneMinusB2 = 1 - b2;
    f32 biasCorrection1 = 1.0f - powf(b1, step);
    f32 biasCorrection2 = 1.0f - powf(b2, step);
    f32 inv_bc1 = 1.0f / biasCorrection1;
    f32 inv_bc2 = 1.0f / biasCorrection2;

    SHAPES_PRAGMA_SIMD
    for (tensor_size_t i = 0; i < trip.param->size; i++) {
      mVals[i] = b1 * mVals[i] + oneMinusB1 * gradVals[i];
      vVals[i] = b2 * vVals[i] + oneMinusB2 * gradVals[i] * gradVals[i];

      f32 mHat = mVals[i] * inv_bc1;
      f32 vHat = vVals[i] * inv_bc2;

      f32 rsqrt = rsqrtf(vHat + epsilon);
      rsqrt *= (1.5f - 0.5f * (vHat + epsilon) * rsqrt * rsqrt);
      pVals[i] -= a * mHat * rsqrt;
    }
  }

  return OK;
}
