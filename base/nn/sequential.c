#include "nn/nn.h"
#include "result/result.h"
#include "shapes.h"
#include "tensor_internal.h"
#include "types.h"
#include "utils_lib/array.h"
#include <stdbool.h>
#include <stddef.h>
#include "nn_internal.h"
#include "utils_lib/memory.h"

typedef struct sequentialModelData {
  Array *layers;
  Array *parameters;
} sequentialModel;

FowardPassOp *shapesnn_Sequential(Context *ctx, FowardPassOp **layerOps, size_t numLayers, Dtype dtype) {
  PANIC_IF(numLayers == 0, ZERO_LAYERS_PASSED);
  PANIC_IF_NULL(layerOps);

  Array *layers = MakeArray(ctx->memory, sizeof(FowardPassOp *), numLayers);
  Array *parameters = shapes_Make_DynamicTensorArray(ctx->memory);

  for (size_t i = 0; i < numLayers; i++) {
    FowardPassOp *op = layerOps[i];

    PANIC_IF(op->dtype != dtype, ERR_DTYPE_MISMATCH);

    bool isFowardPassOp = false;
    u8 counter = 0;
    while (true) {
      if (op->type == LayerOps[counter]) {
        isFowardPassOp = true;
        break;
      } else if (LayerOps[counter] == OP_NONE) {
        break;
      }
      counter += 1;
    }

    PANIC_IF(!isFowardPassOp, NON_LAYER_OP_PASSED);

    array_AppendFowardPassOp(layers, op);

    // TODO: use scratch context here
    Array *params = shapesnn_Parameters(ctx, op);
    for (size_t ip = 0; ip < params->size; ip++) {
      Tensor *p = shapes_Array_TensorIdx(params, ip);
      shapes_Array_AppendTensor(parameters, p);
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
  for (RANGE(i, layers->size)) {
    FowardPassOp *layer = array_FowardPassOpIdx(layers, i);
    out = shapesnn_Forward(ctx, layer, out);
  }

  return out;
}

Array *sequentialModelParameters(Context *ctx, FowardPassOp *modelOp) {
  (void)ctx;
  PANIC_IF(modelOp->type != OP_SEQUENTIAL, OP_NOT_SEQUENTIAL);

  sequentialModel *model = modelOp->op;
  return model->parameters;
}
