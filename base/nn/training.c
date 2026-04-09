#include "nn/nn.h"
#include "result/result.h"
#include "shapes.h"
#include "tensor/tensor_internal.h"
#include "tensor/value.h"

void ZeroGrad(Context *ctx, Array *graph) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(graph == NULL, ERR_NULL_PTR);

  for (size_t i = 0; i < graph->size; i++) {
    Tensor *p = Array_TensorIdx(graph, i);
    Tensor *g = p->grad;

    SetValues(g, VALUE(g->dtype, 0));
  }
}
