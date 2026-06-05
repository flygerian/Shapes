#include "nn.h"
#include "result.h"
#include "shapes.h"
#include "shapes_internal.h"
#include "types.h"
#include "nn_internal.h"
#include "array.h"
#include "olib.h"
#include <string.h>

static Array_NamedTensor wrapNamed(Context *ctx, olib_Array *tensors) {
  Array_NamedTensor named = olib_MakeArray(ctx->memory, sizeof(NamedTensor), tensors->size);
  for (RANGE(i,tensors->size)) {
    Tensor t = shapes_Array_TensorIdx(tensors, i);
    PANIC_IF(t.label == NULL, ERR_NULL_PTR);
    NamedTensor nt = {.name = olib_MakeString(ctx->memory, t.label), .tensor = t};
    olib_ArrayAppend(named, &nt);
  }
  return named;
}

Array_NamedTensor shapesnn_Tensors(Context *ctx, FowardPassOp *op) {
  PANIC_IF(ctx == NULL, ERR_NULL_PTR);
  PANIC_IF(op == NULL, ERR_NULL_PTR);

  switch (op->type) {
    case OP_DENSE: return wrapNamed(ctx, denseLayerTensors(ctx, (Layer *)op->op));
    case OP_EMBEDDING: return wrapNamed(ctx, embeddingLayerTensors(ctx, (Layer *)op->op));
    case OP_BATCH_NORM: return wrapNamed(ctx, batchNormLayerTensors(ctx, (Layer *)op->op));
    case OP_TANH: return wrapNamed(ctx, tanhLayerTensors(ctx, (Layer *)op->op));
    case OP_RELU: return wrapNamed(ctx, reluLayerTensors(ctx, (Layer *)op->op));
    case OP_MAXPOOL2D: return wrapNamed(ctx, maxPool2dLayerTensors(ctx, (Layer *)op->op));
    case OP_ADAPTIVE_AVG_POOL2D: return wrapNamed(ctx, adaptiveAvgPool2dLayerTensors(ctx, (Layer *)op->op));
    case OP_CONV2D: return wrapNamed(ctx, conv2dLayerTensors(ctx, (Layer *)op->op));
    case OP_FLATTEN: return wrapNamed(ctx, flattenLayerTensors(ctx));
    case OP_SEQUENTIAL: return sequentialModelTensors(ctx, op);
    default: PANIC_WITH_CODE(LAYER_OP_NOT_FOUND);
  }
}

void shapesnn_SaveAsSafeTensors(Context *ctx, FowardPassOp *model, string path) {
  PANIC_IF_NULL(model);
  PANIC_IF(model->type != OP_SEQUENTIAL, ERR_NO_OP);

  Array_NamedTensor named = shapesnn_Tensors(ctx, model);
  shapesnn_SafeTensors_Save(ctx, named, path);
}

void shapesnn_Load(Context *ctx, FowardPassOp *op, olib_Array *tensors) {
  PANIC_IF(ctx == NULL, ERR_NULL_PTR);
  PANIC_IF(op == NULL, ERR_NULL_PTR);
  PANIC_IF(tensors == NULL, ERR_NULL_PTR);

  switch (op->type) {
    case OP_DENSE: denseLayerLoad(ctx, (Layer *)op->op, tensors); return;
    case OP_EMBEDDING: embeddingLayerLoad(ctx, (Layer *)op->op, tensors); return;
    case OP_BATCH_NORM: batchNormLayerLoad(ctx, (Layer *)op->op, tensors); return;
    case OP_TANH: tanhLayerLoad(ctx, (Layer *)op->op, tensors); return;
    case OP_RELU: reluLayerLoad(ctx, (Layer *)op->op, tensors); return;
    case OP_MAXPOOL2D: maxPool2dLayerLoad(ctx, (Layer *)op->op, tensors); return;
    case OP_ADAPTIVE_AVG_POOL2D: adaptiveAvgPool2dLayerLoad(ctx, (Layer *)op->op, tensors); return;
    case OP_CONV2D: conv2dLayerLoad(ctx, (Layer *)op->op, tensors); return;
    case OP_FLATTEN: flattenLayerLoad(ctx, tensors); return;
    case OP_SEQUENTIAL: sequentialModelLoad(ctx, op, tensors); return;
    default: PANIC_WITH_CODE(LAYER_OP_NOT_FOUND);
  }
}

void shapesnn_LoadFromSafeTensors(Context *ctx, FowardPassOp *model, string path) {
  PANIC_IF_NULL(model);
  PANIC_IF(model->type != OP_SEQUENTIAL, ERR_NO_OP);

  Array_NamedTensor loaded = shapesnn_SafeTensors_Load(ctx, path);
  Array_NamedTensor expected = shapesnn_Tensors(ctx, model);

  olib_Array *tensors = olib_MakeArray(ctx->memory, sizeof(Tensor), expected->size);
  for (RANGE(i, expected->size)) {
    NamedTensor *expectedNt = (NamedTensor *)olib_ArrayIdx(expected, i);
    Tensor match = {};
    for (RANGE(j, loaded->size)) {
      NamedTensor *loadedNt = (NamedTensor *)olib_ArrayIdx(loaded, j);
      if (strcmp(STR(loadedNt->name), STR(expectedNt->name)) == 0) {
        match = loadedNt->tensor;
        break;
      }
    }
    shapes_Array_AppendTensor(tensors, &match);
  }

  shapesnn_Load(ctx, model, tensors);
}

