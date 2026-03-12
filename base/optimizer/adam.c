#include "common.h"
#include "result/result.h"
#include "tensor/tensor_internal.h"

Result Adam(Context *ctx, AdamData *triplets, size_t numTriplets, f32 b1, f32 b2, size_t step,
            f32 a, f32 epsilon) {
  if (triplets == NULL) {
    return ERR_ADAM_NULL_TRIPLETS;
  }

  for (size_t tripletIdx = 0; tripletIdx < numTriplets; tripletIdx++) {
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

    if (trip.paramGrad == NULL) {
      return ERR_ADAM_NULL_GRAD;
    }

    if (isNotFloatType(trip.m) || isNotFloatType(trip.param) || isNotFloatType(trip.v) ||
        isNotFloatType(trip.paramGrad)) {
      return ERR_ADAM_ONLY_FLOAT_TENSORS;
    }

    if (trip.param->size != trip.m->size || trip.param->size != trip.v->size ||
        trip.param->size != trip.paramGrad->size) {
      return ERR_ADAM_PARAMS_SIZE_MISMATCH;
    }

    f32 *gradVals = trip.paramGrad->values;
    f32 *pVals = trip.param->values;

    f32 *mVals = trip.m->values;
    f32 *vVals = trip.v->values;

    // Compute bias correction factors: (1 - beta^step)
    f32 biasCorrection1 = 1.0f - powf(b1, step);
    f32 biasCorrection2 = 1.0f - powf(b2, step);
    f32 inv_bc1 = 1.0f / biasCorrection1;
    f32 inv_bc2 = 1.0f / biasCorrection2;

    for (tensor_size_t i = 0; i < trip.param->size; i++) {
      mVals[i] = b1 * mVals[i] + (1 - b1) * gradVals[i];
      vVals[i] = b2 * vVals[i] + (1 - b2) * gradVals[i] * gradVals[i];

      f32 mHat = mVals[i] * inv_bc1;
      f32 vHat = vVals[i] * inv_bc2;

      pVals[i] -= a * mHat / (sqrtf(vHat) + epsilon);
    }
  }

  return OK;
}
