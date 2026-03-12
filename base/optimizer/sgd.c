#include "common.h"
#include "memory.h"
#include "result/result.h"
#include "shapes.h"
#include "tensor/tensor_internal.h"
#include <math.h>
#include <signal.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

Result Sgd(Context *ctx, Tensor **parameters, Tensor **parameterGrads, size_t numParameters, f32 learningRate) {
  (void)ctx;
  if (parameters == NULL || parameterGrads == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (learningRate <= 0) {
    return ERR_LEARNING_RATE_CANNOT_BE_ZERO_OR_NEGATIVE;
  }

  for (size_t i = 0; i < numParameters; i++) {
    Tensor *p = parameters[i];
    Tensor *g = parameterGrads[i];

    if (p == NULL || g == NULL) {
      return ERR_NULL_TENSOR_PROVIDED;
    }

    if (p->dtype != g->dtype) {
      return ERR_SGD_PARAMS_GRAD_DTYPE_MISMATCH;
    }

    if (p->size != g->size) {
      return ERR_SGD_PARAMS_NUMBER_MISMATCH;
    }

    if (!p->isContigous || !g->isContigous) {
      return ERR_SGD_PARAMS_HAVE_TO_BE_CONTIGOUS;
    }

    if (isFloatNotType(p) || isFloatNotType(g)) {
      return ERR_SGD_PARAMS_HAVE_TO_BE_FLOAT; 
    }
  }

  for (size_t i = 0; i < numParameters; i++) {
    Tensor *p = parameters[i];
    Tensor *g = parameterGrads[i];

    if (p->dtype == F16) {
      f16 *parameterVals = p->values;
      f16 *gradVals = g->values;
      for (tensor_size_t x = 0; x < p->size; x++) {
        parameterVals[x] -= (gradVals[x] * learningRate);
      }
    }

    if (p->dtype == F32) {
      f32 *parameterVals = p->values;
      f32 *gradVals = g->values;
      for (tensor_size_t x = 0; x < p->size; x++) {
        parameterVals[x] -= (gradVals[x] * learningRate);
      }
    }
    
    if (p->dtype == F64) {
      f64 *parameterVals = p->values;
      f64 *gradVals = g->values;
      for (tensor_size_t x = 0; x < p->size; x++) {
        parameterVals[x] -= (gradVals[x] * learningRate);
      }
    }
  }

  return OK;
}

Result Adam(Context *ctx, AdamTriplet *triplets, size_t numTriplets, f32 b1, f32 b2, size_t step, f32 a, f32 epsilon) {
  if (triplets == NULL) {
    return ERR_ADAM_NULL_TRIPLETS; 
  }

  for (size_t tripletIdx = 0; tripletIdx < numTriplets; tripletIdx++) {
    AdamTriplet trip = triplets[tripletIdx];
    
    if (trip.m == NULL) {
      return  ERR_ADAM_NULL_M; 
    }

    if (trip.v == NULL) {
      return ERR_ADAM_NULL_V;
    }

    if (trip.param == NULL) {
      return ERR_ADAM_NULL_PARAM;
    }

    if (isFloatNotType(trip.m) || isFloatNotType(trip.param) || isFloatNotType(trip.v)) {
      return ERR_ADAM_ONLY_FLOAT_TENSORS;
    }


    f32 *gradVals = trip.paramGrad->values;
    f32 *pVals = trip.param->values;

    f32 *mVals = trip.m->values;
    f32 *vVals = trip.v->values;

    f32 b1XStep = powf(b1, step);
    f32 b2XStep = powf(b2, step);

    for (tensor_size_t i = 0; i < trip.param->size; i++) {
      mVals[i] = b1 * mVals[i] + (1 - b1) * gradVals[i];
      vVals[i] = b2 * vVals[i] + (1 - b2) * gradVals[i] * gradVals[i];

      f32 mHat = mVals[i] / (1 - b1XStep);
      f32 vHat = vVals[i] / (1 - b2XStep);

      pVals[i] -= a * mHat / (sqrtf(vHat) + epsilon);
    }

  }
}
