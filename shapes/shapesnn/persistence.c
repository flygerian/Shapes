#include "nn.h"
#include "result.h"
#include "shapes.h"
#include "types.h"
#include "nn_internal.h"
#include "array.h"
#include "olib.h"
#include <string.h>

static shapes_ArrayNamedTensor wrapNamed(shapes_Context *ctx, olib_Array *tensors) {
  shapes_ArrayNamedTensor named = olib_MakeArray(ctx->memory, sizeof(shapesnn_NamedTensor), tensors->size);
  for (RANGE(i,tensors->size)) {
    shapes_Tensor t = shapes_ArrayTensorIdx(tensors, i);
    PANIC_IF(t.label == NULL, ERR_NULL_PTR);
    shapesnn_NamedTensor nt = {.name = olib_MakeString(ctx->memory, t.label), .tensor = t};
    olib_ArrayAppend(named, &nt);
  }
  return named;
}

shapes_ArrayNamedTensor shapesnn_Tensors(shapes_Context *ctx, shapesnn_FowardPassOp *op) {
  PANIC_IF(ctx == NULL, ERR_NULL_PTR);
  PANIC_IF(op == NULL, ERR_NULL_PTR);

  switch (op->type) {
    case OP_DENSE: return wrapNamed(ctx, denseLayerTensors(ctx, (shapesnn_layer *)op->op));
    case OP_EMBEDDING: return wrapNamed(ctx, embeddingLayerTensors(ctx, (shapesnn_layer *)op->op));
    case OP_BATCH_NORM: return wrapNamed(ctx, batchNormLayerTensors(ctx, (shapesnn_layer *)op->op));
    case OP_TANH: return wrapNamed(ctx, tanhLayerTensors(ctx, (shapesnn_layer *)op->op));
    case OP_RELU: return wrapNamed(ctx, reluLayerTensors(ctx, (shapesnn_layer *)op->op));
    case OP_MAXPOOL2D: return wrapNamed(ctx, maxPool2dLayerTensors(ctx, (shapesnn_layer *)op->op));
    case OP_ADAPTIVE_AVG_POOL2D: return wrapNamed(ctx, adaptiveAvgPool2dLayerTensors(ctx, (shapesnn_layer *)op->op));
    case OP_CONV2D: return wrapNamed(ctx, conv2dLayerTensors(ctx, (shapesnn_layer *)op->op));
    case OP_FLATTEN: return wrapNamed(ctx, flattenLayerTensors(ctx));
    case OP_SEQUENTIAL: return sequentialModelTensors(ctx, op);
    default: PANIC_WITH_CODE(LAYER_OP_NOT_FOUND);
  }
}

void shapesnn_SaveAsSafeTensors(shapes_Context *ctx, shapesnn_FowardPassOp *model, string path) {
  PANIC_IF_NULL(model);
  PANIC_IF(model->type != OP_SEQUENTIAL, ERR_NO_OP);

  shapes_ArrayNamedTensor named = shapesnn_Tensors(ctx, model);
  shapesnn_SafeTensors_Save(ctx, named, path);
}

void shapesnn_Load(shapes_Context *ctx, shapesnn_FowardPassOp *op, olib_Array *tensors) {
  PANIC_IF(ctx == NULL, ERR_NULL_PTR);
  PANIC_IF(op == NULL, ERR_NULL_PTR);
  PANIC_IF(tensors == NULL, ERR_NULL_PTR);

  switch (op->type) {
    case OP_DENSE: denseLayerLoad(ctx, (shapesnn_layer *)op->op, tensors); return;
    case OP_EMBEDDING: embeddingLayerLoad(ctx, (shapesnn_layer *)op->op, tensors); return;
    case OP_BATCH_NORM: batchNormLayerLoad(ctx, (shapesnn_layer *)op->op, tensors); return;
    case OP_TANH: tanhLayerLoad(ctx, (shapesnn_layer *)op->op, tensors); return;
    case OP_RELU: reluLayerLoad(ctx, (shapesnn_layer *)op->op, tensors); return;
    case OP_MAXPOOL2D: maxPool2dLayerLoad(ctx, (shapesnn_layer *)op->op, tensors); return;
    case OP_ADAPTIVE_AVG_POOL2D: adaptiveAvgPool2dLayerLoad(ctx, (shapesnn_layer *)op->op, tensors); return;
    case OP_CONV2D: conv2dLayerLoad(ctx, (shapesnn_layer *)op->op, tensors); return;
    case OP_FLATTEN: flattenLayerLoad(ctx, tensors); return;
    case OP_SEQUENTIAL: sequentialModelLoad(ctx, op, tensors); return;
    default: PANIC_WITH_CODE(LAYER_OP_NOT_FOUND);
  }
}

void shapesnn_LoadFromSafeTensors(shapes_Context *ctx, shapesnn_FowardPassOp *model, string path) {
  PANIC_IF_NULL(model);
  PANIC_IF(model->type != OP_SEQUENTIAL, ERR_NO_OP);

  shapes_ArrayNamedTensor loaded = shapesnn_SafeTensors_Load(ctx, path);
  shapes_ArrayNamedTensor expected = shapesnn_Tensors(ctx, model);

  olib_Array *tensors = olib_MakeArray(ctx->memory, sizeof(shapes_Tensor), expected->size);
  for (RANGE(i, expected->size)) {
    shapesnn_NamedTensor *expectedNt = (shapesnn_NamedTensor *)olib_ArrayIdx(expected, i);
    shapes_Tensor match = {};
    for (RANGE(j, loaded->size)) {
      shapesnn_NamedTensor *loadedNt = (shapesnn_NamedTensor *)olib_ArrayIdx(loaded, j);
      if (strcmp(STR(loadedNt->name), STR(expectedNt->name)) == 0) {
        match = loadedNt->tensor;
        break;
      }
    }
    shapes_ArrayAppendTensor(tensors, &match);
  }

  shapesnn_Load(ctx, model, tensors);
}

