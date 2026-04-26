#include "nn/nn.h"
#include "result/result.h"
#include "tensor/tensor_internal.h"
#include "tensor/types.h"
#include "utils_lib/array.h"
#include <stdbool.h>
#include <stddef.h>
#include "nn_internal.h"
#include "utils_lib/memory.h"

typedef struct sequentialModelData {
  Array *layers;
  Array *parameters;
} sequentialModel;

FowardPassOp *Make_Sequential(Context *ctx, FowardPassOp **layerOps, size_t numLayers,
                              Dtype dtype) {
  PANIC_IF(numLayers == 0, ZERO_LAYERS_PASSED);
  PANIC_IF_NULL(layerOps);

  Array *layers = MakeArray(ctx->memory, sizeof(FowardPassOp *), numLayers);
  Array *parameters = MakeDynamicArray(ctx->memory, sizeof(Tensor));

  for (size_t i = 0; i < numLayers; i++) {
    FowardPassOp *op = layerOps[i];

    PANIC_IF(op->dtype != dtype, ERR_DTYPE_MISMATCH);

    bool isFowardPassOp = false;
    for (size_t j = 0; j < NUM_LAYER_OPS; j++) {
      if (op->type == LayerOps[j]) {
        isFowardPassOp = true;
        break;
      }
    }

    PANIC_IF(!isFowardPassOp, NON_LAYER_OP_PASSED);

    array_AppendFowardPassOp(layers, op);

    // TODO: use scratch context here
    Array *params = Parameters(ctx, op);
    for (size_t ip = 0; ip < params->size; ip++) {
      Tensor *p = Array_TensorIdx(params, ip);
      Array_AppendTensor(parameters, p);
    }
  }

  sequentialModel *model = allocate(ctx->memory, sizeof(sequentialModel));
  *model = (sequentialModel){.layers = layers, .parameters = parameters};

  FowardPassOp *op = allocate(ctx->memory, sizeof(FowardPassOp));
  *op = (FowardPassOp){.type = OP_SEQUENTIAL, .op = model, .ctx = ctx, .dtype = dtype};
  return op;
}

Tensor *sequentialModelForward(Context *ctx, FowardPassOp *modelOp, Tensor *input) {
  PANIC_IF(modelOp->type != OP_SEQUENTIAL, OP_NOT_SEQUENTIAL);

  sequentialModel *model = modelOp->op;
  Array *layers = model->layers;

  Tensor *out = input;
  for (size_t i = 0; i < layers->size; i++) {
    FowardPassOp *layer = array_FowardPassOpIdx(layers, i);
    out = Forward(ctx, layer, out);
  }

  return out;
}

Array *sequentialModelParameters(Context *ctx, FowardPassOp *modelOp) {
  (void)ctx;
  PANIC_IF(modelOp->type != OP_SEQUENTIAL, OP_NOT_SEQUENTIAL);

  sequentialModel *model = modelOp->op;
  return model->parameters;
}
