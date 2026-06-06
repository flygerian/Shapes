#include "nn.h"
#include "result.h"
#include "shapes.h"
#include "types.h"
#include "array.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "nn_internal.h"
#include "memory.h"
#include "olib.h"

#define MAX_EXPECTED_NUM_LAYER_PARAMS 5

shapesnn_FowardPassOp shapesnn_Sequential(shapes_Context *ctx, shapesnn_FowardPassOp *layerOps, size_t numLayers, shapes_Dtype dtype) {
  PANIC_IF(numLayers == 0, ZERO_LAYERS_PASSED);
  PANIC_IF_NULL(layerOps);

  olib_Array *layers = olib_MakeArray(ctx->memory, sizeof(shapesnn_FowardPassOp), numLayers);
  olib_Array *parameters = shapes_MakeTensorArray(ctx->memory, MAX_EXPECTED_NUM_LAYER_PARAMS * numLayers);

  for (size_t i = 0; i < numLayers; i++) {
    shapesnn_FowardPassOp op = layerOps[i];

    PANIC_IF(op.dtype != dtype, ERR_DTYPE_MISMATCH);

    bool isFowardPassOp = false;
    u8 counter = 0;
    while (true) {
      if (op.type == LayerOps[counter]) {
        isFowardPassOp = true;
        break;
      } else if (LayerOps[counter] == OP_NONE) {
        break;
      }
      counter += 1;
    }

    PANIC_IF(!isFowardPassOp, NON_LAYER_OP_PASSED);

    array_AppendFowardPassOp(layers, &op);

    olib_Array *params = shapesnn_Parameters(ctx, &op);
    for (RANGE(ip, params->size)) {
      shapes_Tensor p = shapes_ArrayTensorIdx(params, ip);
      shapes_ArrayAppendTensor(parameters, &p);
    }
  }

  sequentialModel *model = olib_Allocate(ctx->memory, sizeof(sequentialModel));
  *model = (sequentialModel){.layers = layers, .parameters = parameters};

  return (shapesnn_FowardPassOp){.type = OP_SEQUENTIAL, .op = model, .ctx = ctx, .dtype = dtype};
}

shapes_Tensor sequentialModelForward(shapes_Context *ctx, shapesnn_FowardPassOp *modelOp, shapes_Tensor *input) {
  PANIC_IF(modelOp->type != OP_SEQUENTIAL, OP_NOT_SEQUENTIAL);

  sequentialModel *model = modelOp->op;
  olib_Array *layers = model->layers;

  shapes_Tensor out = *input;
  for (RANGE(i, layers->size)) {
    shapesnn_FowardPassOp layer = array_FowardPassOpIdx(layers, i);
    out = shapesnn_Forward(ctx, &layer, &out);
  }

  return out;
}

olib_Array *sequentialModelParameters(shapes_Context *ctx, shapesnn_FowardPassOp *modelOp) {
  (void)ctx;
  PANIC_IF(modelOp->type != OP_SEQUENTIAL, OP_NOT_SEQUENTIAL);

  sequentialModel *model = modelOp->op;
  return model->parameters;
}

void sequentialModelLoad(shapes_Context *ctx, shapesnn_FowardPassOp *modelOp, olib_Array *tensors) {
  PANIC_IF(modelOp->type != OP_SEQUENTIAL, OP_NOT_SEQUENTIAL);
  PANIC_IF(tensors->elemSize != sizeof(shapes_Tensor), ARRAY_ELEM_SIZE_MISMATCH);

  sequentialModel *model = modelOp->op;
  olib_Array *layers = model->layers;

  size_t cursor = 0;
  for (RANGE(i, layers->size)) {
    shapesnn_FowardPassOp child = array_FowardPassOpIdx(layers, i);
    size_t childCount = shapesnn_Tensors(ctx, &child)->size;

    olib_Array *slice = olib_ArraySlice(tensors, cursor, childCount);
    shapesnn_Load(ctx, &child, slice);
    cursor += childCount;
  }

  PANIC_IF(cursor != tensors->size, ERR_DIM_MISMATCH);
}

olib_Array *sequentialModelTensors(shapes_Context *ctx, shapesnn_FowardPassOp *modelOp) {
  PANIC_IF(modelOp->type != OP_SEQUENTIAL, OP_NOT_SEQUENTIAL);

  sequentialModel *model = modelOp->op;
  olib_Array *layers = model->layers;
  olib_Array *out = olib_MakeDynamicArray(ctx->memory, sizeof(shapesnn_NamedTensor));

  for (RANGE(i, layers->size)) {
    shapesnn_FowardPassOp child = array_FowardPassOpIdx(layers, i);
    olib_Array *childTensors = shapesnn_Tensors(ctx, &child);

    for (RANGE(j, childTensors->size)) {
      shapesnn_NamedTensor *child_nt = (shapesnn_NamedTensor *)olib_ArrayIdx(childTensors, j);
      PANIC_IF(child_nt->name == NULL, ERR_NULL_PTR);

      char buf[256];
      int written = snprintf(buf, sizeof(buf), "layer.%zu.%s", i, STR(child_nt->name));
      PANIC_IF(written < 0 || (size_t)written >= sizeof(buf), ERR_OUT_OF_BOUNDS);

      shapesnn_NamedTensor nt = {
          .name = olib_MakeStringN(ctx->memory, buf, (size_t)written),
          .tensor = child_nt->tensor,
      };
      olib_ArrayAppend(out, &nt);
    }
  }

  return out;
}
