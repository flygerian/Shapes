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

Result Sgd(Context *ctx, Tensor **parameters, Tensor **parameterGrads, size_t numParameters,
           f32 learningRate) {
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

    if (isNotFloatType(p) || isNotFloatType(g)) {
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
