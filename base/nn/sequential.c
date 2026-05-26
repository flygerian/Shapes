#include "nn/nn.h"
#include "result/result.h"
#include "shapes.h"
#include "tensor_internal.h"
#include "types.h"
#include "utils_lib/array.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "nn_internal.h"
#include "utils_lib/memory.h"


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

void sequentialModelLoad(Context *ctx, FowardPassOp *modelOp, Array *tensors) {
  PANIC_IF(modelOp->type != OP_SEQUENTIAL, OP_NOT_SEQUENTIAL);
  PANIC_IF(tensors->elemSize != sizeof(Tensor *), ARRAY_ELEM_SIZE_MISMATCH);

  sequentialModel *model = modelOp->op;
  Array *layers = model->layers;

  size_t cursor = 0;
  for (RANGE(i, layers->size)) {
    FowardPassOp *child = array_FowardPassOpIdx(layers, i);
    size_t childCount = shapesnn_Tensors(ctx, child)->size;

    Array *slice = Array_Slice(tensors, cursor, childCount);
    shapesnn_Load(ctx, child, slice);
    cursor += childCount;
  }

  PANIC_IF(cursor != tensors->size, ERR_DIM_MISMATCH);
}

Array *sequentialModelTensors(Context *ctx, FowardPassOp *modelOp) {
  PANIC_IF(modelOp->type != OP_SEQUENTIAL, OP_NOT_SEQUENTIAL);

  sequentialModel *model = modelOp->op;
  Array *layers = model->layers;
  Array *out = MakeDynamicArray(ctx->memory, sizeof(NamedTensor));

  for (RANGE(i, layers->size)) {
    FowardPassOp *child = array_FowardPassOpIdx(layers, i);
    Array *childTensors = shapesnn_Tensors(ctx, child);

    for (RANGE(j, childTensors->size)) {
      NamedTensor *child_nt = (NamedTensor *)Array_Idx(childTensors, j);
      PANIC_IF(child_nt->name == NULL, ERR_NULL_PTR);

      char buf[256];
      int written = snprintf(buf, sizeof(buf), "layer.%zu.%s", i, STR(child_nt->name));
      PANIC_IF(written < 0 || (size_t)written >= sizeof(buf), ERR_OUT_OF_BOUNDS);

      NamedTensor nt = {
          .name = MakeStringN(ctx->memory, buf, (size_t)written),
          .tensor = child_nt->tensor,
      };
      Array_Append(out, &nt);
    }
  }

  return out;
}
