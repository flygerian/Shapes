#include "nn/nn.h"
#include "result/result.h"
#include "shapes.h"
#include "tensor/value.h"
#include <stdlib.h>

void ZeroGrad(Context *ctx, Array *graph) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(graph == NULL, ERR_NULL_PTR);

  for (size_t i = 0; i < graph->size; i++) {
    Tensor *p = *(Tensor **)Array_Idx(graph, i);
    Tensor *g = p->grad;

    SetValues(g, VALUE(g->dtype, 0));
  }
}
