#include "result.h"
#include "binary_op_helpers.h"
#include <math.h>

#ifndef rsqrtf
static inline float rsqrtf(float x) {
  return 1.0f / sqrtf(x);
}
#endif

static Result adamF32(AdamData *triplets, size_t numParameters, f32 b1, f32 b2, size_t step, f32 a, f32 epsilon) {
  f32 oneMinusB1 = 1.0f - b1;
  f32 oneMinusB2 = 1.0f - b2;
  f32 inv_bc1 = 1.0f / (1.0f - powf(b1, step));
  f32 inv_bc2 = 1.0f / (1.0f - powf(b2, step));

  for (size_t t = 0; t < numParameters; t++) {
    AdamData *trip = &triplets[t];
    f32 *pVals = trip->param;
    f32 *gVals = trip->grad;
    f32 *mVals = trip->m;
    f32 *vVals = trip->v;

    SHAPES_PRAGMA_SIMD
    for (tensor_size_t i = 0; i < trip->size; i++) {
      mVals[i] = b1 * mVals[i] + oneMinusB1 * gVals[i];
      vVals[i] = b2 * vVals[i] + oneMinusB2 * gVals[i] * gVals[i];

      f32 mHat = mVals[i] * inv_bc1;
      f32 vHat = vVals[i] * inv_bc2;

      f32 rsqrt = rsqrtf(vHat + epsilon);
      rsqrt *= (1.5f - 0.5f * (vHat + epsilon) * rsqrt * rsqrt);
      pVals[i] -= a * mHat * rsqrt;
    }
  }

  return OK;
}

static Result adamF64(AdamData *triplets, size_t numParameters, f64 b1, f64 b2, size_t step, f64 a, f64 epsilon) {
  f64 oneMinusB1 = 1.0 - b1;
  f64 oneMinusB2 = 1.0 - b2;
  f64 inv_bc1 = 1.0 / (1.0 - pow(b1, step));
  f64 inv_bc2 = 1.0 / (1.0 - pow(b2, step));

  for (size_t t = 0; t < numParameters; t++) {
    AdamData *trip = &triplets[t];
    f64 *pVals = trip->param;
    f64 *gVals = trip->grad;
    f64 *mVals = trip->m;
    f64 *vVals = trip->v;

    SHAPES_PRAGMA_SIMD
    for (tensor_size_t i = 0; i < trip->size; i++) {
      mVals[i] = b1 * mVals[i] + oneMinusB1 * gVals[i];
      vVals[i] = b2 * vVals[i] + oneMinusB2 * gVals[i] * gVals[i];

      f64 mHat = mVals[i] * inv_bc1;
      f64 vHat = vVals[i] * inv_bc2;

      f64 rsqrt = 1.0 / sqrt(vHat + epsilon);
      rsqrt *= (1.5 - 0.5 * (vHat + epsilon) * rsqrt * rsqrt);
      pVals[i] -= a * mHat * rsqrt;
    }
  }

  return OK;
}

static bool isFloatDtype(shapes_Dtype dt) {
  return dt == F16 || dt == F32 || dt == F64;
}

Result shapes_optimizer_Adam(Context *ctx, AdamData *triplets, size_t numParameters, f32 b1, f32 b2, size_t step, f32 a, f32 epsilon) {
  (void)ctx;

  if (triplets == NULL) {
    return ERR_ADAM_NULL_TRIPLETS;
  }

  for (size_t tripletIdx = 0; tripletIdx < numParameters; tripletIdx++) {
    AdamData *trip = &triplets[tripletIdx];

    if (trip->param == NULL) {
      return ERR_ADAM_NULL_PARAM;
    }

    if (trip->grad == NULL) {
      return ERR_ADAM_NULL_GRAD;
    }

    if (trip->m == NULL) {
      return ERR_ADAM_NULL_M;
    }

    if (trip->v == NULL) {
      return ERR_ADAM_NULL_V;
    }

    if (!isFloatDtype(trip->dtype)) {
      return ERR_ADAM_ONLY_FLOAT_TENSORS;
    }

    if (trip->size == 0) {
      return ERR_ADAM_PARAMS_SIZE_MISMATCH;
    }
  }

  switch (triplets[0].dtype) {
    case F16:
    case F32: return adamF32(triplets, numParameters, b1, b2, step, a, epsilon);
    case F64: return adamF64(triplets, numParameters, (f64)b1, (f64)b2, step, (f64)a, (f64)epsilon);
    default: return ERR_ADAM_ONLY_FLOAT_TENSORS;
  }
}
