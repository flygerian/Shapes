#include "nn.h"
#include "result.h"
#include "shapes.h"
#include "olib.h"
#include "types.h"
#include <stddef.h>

#define D_MODEL 512
#define NUM_HEADS 4

shapes_Tensor attention(shapes_Context *hostCtx, shapes_Tensor *q, shapes_Tensor *k, shapes_Tensor *v, shapes_Tensor *dk) {
  shapes_Tensor kT = shapes_Transpose(hostCtx, k, 1, 0);
  shapes_Tensor qkT = shapes_MatMul(hostCtx, q, &kT);
  shapes_Tensor sqrtDk = shapes_Sqrt(hostCtx, dk);
  shapes_Tensor qkT_Dk = shapes_Divide(hostCtx, &qkT, &sqrtDk);

  shapes_Tensor qkT_Dk_Softmax = shapesnn_Softmax(hostCtx, &qkT_Dk);
  shapes_Tensor attention = shapes_MatMul(hostCtx, &qkT_Dk_Softmax, v);

  return attention;
}

shapes_Tensor multiheadAttention(shapes_Context *hostCtx, shapes_Tensor *input) {
  f32 initVal = (5.0f / 3.0f) / powf((f32)D_MODEL, 0.5f);
  shapes_Tensor wQ = shapes_MakeRandomTensor(hostCtx, SHAPE2D(D_MODEL, D_MODEL), -initVal, initVal, F32);
  shapes_Tensor wK = shapes_MakeRandomTensor(hostCtx, SHAPE2D(D_MODEL, D_MODEL), -initVal, initVal, F32);
  shapes_Tensor wV = shapes_MakeRandomTensor(hostCtx, SHAPE2D(D_MODEL, D_MODEL), -initVal, initVal, F32);
  shapes_Tensor w0 = shapes_MakeRandomTensor(hostCtx, SHAPE2D(D_MODEL, D_MODEL), -initVal, initVal, F32);

  shapes_Tensor qPrime = shapes_MatMul(hostCtx, input, &wQ);
  shapes_Tensor kPrime = shapes_MatMul(hostCtx, input, &wK);
  shapes_Tensor vPrime = shapes_MatMul(hostCtx, input, &wV);

  shapes_ArrayTensor Q = shapes_SplitAtDim(hostCtx, &qPrime, 1, NUM_HEADS);
  shapes_ArrayTensor K = shapes_SplitAtDim(hostCtx, &kPrime, 1, NUM_HEADS);
  shapes_ArrayTensor V = shapes_SplitAtDim(hostCtx, &vPrime, 1, NUM_HEADS);

  PANIC_IF(Q->size != NUM_HEADS, ERR_DIM_MISMATCH);

  shapes_ArrayTensor dvs = shapes_MakeTensorArray(hostCtx->memory, NUM_HEADS);

  shapes_Tensor embSize = shapes_MakeFloatTensor(hostCtx, SCALAR, (f32) (D_MODEL / NUM_HEADS)); 
  for (RANGE(i, NUM_HEADS)) {
    shapes_Tensor qi = shapes_ArrayTensorIdx(Q, i); 
    shapes_Tensor ki = shapes_ArrayTensorIdx(K, i); 
    shapes_Tensor vi = shapes_ArrayTensorIdx(V, i); 

    shapes_Tensor dvi = attention(hostCtx, &qi, &ki, &vi, &embSize);
    shapes_ArrayAppendTensor(dvs, &dvi);
  }

  shapes_Tensor dConcat = shapes_Concat(hostCtx, dvs, 1);
  shapes_Tensor multiHeadAttn = shapes_MatMul(hostCtx, &dConcat, &w0);
  shapes_PrintTensor(&multiHeadAttn);

  return multiHeadAttn;
}

int main() {
  shapes_Context hostCtx = shapes_InitializeHostContext(25 * GB, 1);
  shapesnn_FowardPassOp layer = shapesnn_Embedding(&hostCtx, F32, 6, D_MODEL);

  i8 indicesArr[3] = {1,3,5};
  shapes_Tensor indices = shapes_MakeFromContigousArray(&hostCtx, SHAPE1D(3), indicesArr, I8);
  shapes_Tensor q = shapesnn_Forward(&hostCtx, &layer, &indices);

  multiheadAttention(&hostCtx, &q);

  return 0;
}
