#include "common.h"
#include "result/result.h"
#include "shapes.h"
#include "tensor/tensor_internal.h"
#include "utils_lib/array.h"
#include <stddef.h>
#include <stdlib.h>

Result Sgd(Context *ctx, Array *parameters, f32 learningRate) {
  PANIC_IF(ctx == NULL || parameters == NULL, ERR_NULL_TENSOR_PROVIDED); 
  PANIC_IF(learningRate <= 0, ERR_LEARNING_RATE_CANNOT_BE_ZERO_OR_NEGATIVE); 

  for (size_t i = 0; i < parameters->size; i++) {
    Tensor *p = *(Tensor **)Array_Idx(parameters, i);
    Tensor *g = p->grad;

    if (p == NULL || g == NULL) {
      return ERR_NULL_TENSOR_PROVIDED;
    }

    if (p->dtype != g->dtype) {
      return ERR_SGD_PARAMS_GRAD_DTYPE_MISMATCH;
    }

    if (p->size != g->size) {
      return ERR_SGD_PARAMS_NUMBER_MISMATCH;
    }

    if (isNotFloatType(p) || isNotFloatType(g)) {
      return ERR_SGD_PARAMS_HAVE_TO_BE_FLOAT;
    }
  }

  for (size_t i = 0; i < parameters->size; i++) {
    Tensor *p = *(Tensor **)Array_Idx(parameters, i);
    Tensor *g = p->grad;

    Tensor *pWork = materializeTensorOnContext(ctx, p);
    Tensor *gWork = materializeTensorOnContext(ctx, g);

    if (ctx->device != NULL && ctx->device->type == CUDA) {
      Result res =
          runCudaSgd(ctx, pWork->dtype, pWork->values, gWork->values, pWork->size, learningRate);
      PANIC_IF(res != OK, res);
    } else {
      if (p->dtype == F16) {
        f16 *parameterVals = pWork->values;
        f16 *gradVals = gWork->values;
        for (tensor_size_t x = 0; x < pWork->size; x++) {
          parameterVals[x] -= (gradVals[x] * learningRate);
        }
      }

      if (p->dtype == F32) {
        f32 *parameterVals = pWork->values;
        f32 *gradVals = gWork->values;
        for (tensor_size_t x = 0; x < pWork->size; x++) {
          parameterVals[x] -= (gradVals[x] * learningRate);
        }
      }

      if (p->dtype == F64) {
        f64 *parameterVals = pWork->values;
        f64 *gradVals = gWork->values;
        for (tensor_size_t x = 0; x < pWork->size; x++) {
          parameterVals[x] -= (gradVals[x] * learningRate);
        }
      }
    }

    freeIfContingousCopy(ctx, pWork);
    freeIfContingousCopy(ctx, gWork);
  }

  return OK;
}
