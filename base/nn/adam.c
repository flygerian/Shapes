#include "nn/nn.h"
#include "result/result.h"
#include "shapes.h"
#include "utils_lib/array.h"
#include "utils_lib/map.h"
#include <stddef.h>
#include <stdint.h>

typedef struct adamState {
  PtrMap *m;
  PtrMap *v;
  f32 b1;
  f32 b2;
  f32 step;
  f32 episolon;
} adamState;

void adamStep(Context *ctx, Optimizer *opts, Array *parameters) {
  adamState *state = opts->state;
  PtrMap *m = state->m;
  PtrMap *v = state->v;

  PANIC_IF(parameters->size == 0, ERR_DIM_MISMATCH);

  for (size_t i = 0; i < parameters->size; i++) {
    Tensor *p = shapes_Array_TensorIdx(parameters, i);
    if (!PtrMap_Contains(m, p)) {
      PtrMap_Put(m, p, shapes_Make_ZerosTensor(ctx, p->shape));
    }

    if (!PtrMap_Contains(v, p)) {
      PtrMap_Put(v, p, shapes_Make_ZerosTensor(ctx, p->shape));
    }
  }

  AdamData triplets[parameters->size];

  for (size_t i = 0; i < parameters->size; i++) {
    Tensor *p = shapes_Array_TensorIdx(parameters, i);
    triplets[i] = (AdamData){
        .m = ((Tensor *)PtrMap_Get(m, p))->values,
        .v = ((Tensor *)PtrMap_Get(v, p))->values,
        .param = p->values,
        .grad = p->grad->values,
        .size = p->size,
        .dtype = p->dtype,
    };
  }

  state->step++;
  Result res = shapes_optimizer_Adam(ctx, triplets, parameters->size, state->b1, state->b2, state->step, opts->learningRate, state->episolon);

  PANIC_IF(res != OK, res);
}

Optimizer *optimizer_Adam(Context *ctx, f32 learningRate) {
  adamState state = {
      .episolon = 1e-8,
      .m = Make_PtrSet(ctx->memory),
      .v = Make_PtrSet(ctx->memory),
      .b1 = 0.9,
      .b2 = 0.999,
      .step = 0,
  };

  adamState *aState = allocate(ctx->memory, sizeof(adamState));
  *aState = state;

  Optimizer *opt = allocate(ctx->memory, sizeof(Optimizer));
  *opt = (Optimizer){.learningRate = learningRate, .state = aState, .opType = OP_ADAM};
  return opt;
}
