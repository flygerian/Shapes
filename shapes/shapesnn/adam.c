#include "nn.h"
#include "result.h"
#include "shapes.h"
#include "array.h"
#include "map.h"
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

void adamStep(shapes_Context *ctx, shapesnn_Optimizer *opts, olib_Array *parameters) {
  adamState *state = opts->state;
  PtrMap *m = state->m;
  PtrMap *v = state->v;

  PANIC_IF(parameters->size == 0, ERR_DIM_MISMATCH);

  for (size_t i = 0; i < parameters->size; i++) {
    Tensor *p = shapes_ArrayTensorPtrIdx(parameters, i);
    if (!PtrMap_Contains(m, &p)) {
      Tensor *mEntry = olib_Allocate(ctx->memory, sizeof(Tensor));
      PANIC_IF(mEntry == NULL, ALLOCATION_FAILED);
      *mEntry = shapes_MakeZerosTensor(ctx, p->shape);
      PtrMap_Put(m, (void*) p, mEntry);
    }

    if (!PtrMap_Contains(v, p)) {
      Tensor *vEntry = olib_Allocate(ctx->memory, sizeof(Tensor));
      PANIC_IF(vEntry == NULL, ALLOCATION_FAILED);
      *vEntry = shapes_MakeZerosTensor(ctx, p->shape);
      PtrMap_Put(v, p, vEntry);
    }
  }

  shapes_AdamData triplets[parameters->size];

  for (size_t i = 0; i < parameters->size; i++) {
    Tensor *p = shapes_ArrayTensorPtrIdx(parameters, i);
    triplets[i] = (shapes_AdamData){
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

shapesnn_Optimizer shapesnn_Adam(shapes_Context *ctx, f32 learningRate) {
  adamState state = {
      .episolon = 1e-8,
      .m = Make_PtrSet(ctx->memory),
      .v = Make_PtrSet(ctx->memory),
      .b1 = 0.9,
      .b2 = 0.999,
      .step = 0,
  };

  adamState *aState = olib_Allocate(ctx->memory, sizeof(adamState));
  *aState = state;

  return (shapesnn_Optimizer){.learningRate = learningRate, .state = aState, .opType = OP_ADAM};
}
