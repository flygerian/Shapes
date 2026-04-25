#include "common.h"
#include "nn/nn.h"
#include "result/result.h"
#include "shapes.h"
#include "tensor/tensor_internal.h"
#include "tensor/value.h"
#include "utils_lib/array.h"
#include "utils_lib/bitset.h"
#include "utils_lib/memory.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BATCH_SIZE 32
#define NUM_EPOCHS 5

Array *getNames(Context *ctx) {
  FILE *f = fopen("names.txt", "r");
  PANIC_IF(!f, FILE_OPEN_FAILED);

  Array *words = MakeArray(ctx->memory, sizeof(String), 33000);

  char *line = NULL;
  size_t cap = 0;
  size_t len;
  while ((len = getline(&line, &cap, f)) != -1) {
    line[strcspn(line, "\n")] = '\0';
    String l = MakeString(ctx->memory, line);
    Array_AppendString(words, l);
  }

  fclose(f);
  return words;
}

Bitset *MakeCharSet(Context *ctx, Array *words) {
  Bitset *charset = Make_Bitset(ctx->memory);

  for (size_t i = 0; i < words->size; i++) {
    String word = Array_StringIdx(words, i);
    char *chars = (char *)word->items;
    for (size_t j = 0; j < word->size; j++) {
      Bitset_Put(charset, (unsigned char)chars[j]);
    }
  }

  return charset;
}

Array *MakeStoi(Context *ctx, Bitset *charset) {
  Array *stoi = MakeArray(ctx->memory, sizeof(int), 256);

  for (size_t i = 0; i < 256; i++) {
    int val = -1;
    Array_SetAt(stoi, i, &val);
  }

  int dotVal = 0;
  Array_SetAt(stoi, '.', &dotVal);

  int idx = 1;
  for (char c = 'a'; c <= 'z'; c++) {
    if (Bitset_Contains(charset, (unsigned char)c)) {
      Array_SetAt(stoi, (size_t)c, &idx);
      idx++;
    }
  }

  stoi->size = 256;
  return stoi;
}

Array *MakeItos(Context *ctx, Bitset *charset) {
  size_t vocabSize = 1;
  for (char c = 'a'; c <= 'z'; c++) {
    if (Bitset_Contains(charset, (unsigned char)c)) {
      vocabSize++;
    }
  }

  Array *itos = MakeArray(ctx->memory, sizeof(char), vocabSize);

  char dot = '.';
  Array_SetAt(itos, 0, &dot);

  size_t idx = 1;
  for (char c = 'a'; c <= 'z'; c++) {
    if (Bitset_Contains(charset, (unsigned char)c)) {
      Array_SetAt(itos, idx, &c);
      idx++;
    }
  }

  itos->size = vocabSize;
  return itos;
}

int Array_StoiGet(Array *stoi, char c) {
  int *items = (int *)stoi->items;
  return items[(unsigned char)c];
}

char Array_ItosGet(Array *itos, int idx) {
  char *items = (char *)itos->items;
  return items[idx];
}

typedef struct {
  int context[3];
  int target;
} DatasetPair;

typedef struct {
  Array *inputs;
  Array *targets;
  size_t numBatches;
} BatchedDataset;

Array *BuildDataset(Context *ctx, Array *words, Array *stoi) {
  size_t totalPairs = 0;
  for (size_t w = 0; w < words->size; w++) {
    String word = Array_StringIdx(words, w);
    totalPairs += word->size + 1;
  }

  Array *dataset = MakeArray(ctx->memory, sizeof(DatasetPair), totalPairs);

  for (size_t w = 0; w < words->size; w++) {
    String word = Array_StringIdx(words, w);
    char *chars = (char *)word->items;
    size_t wordLen = word->size;

    char padded[wordLen + 5];
    padded[0] = '.';
    padded[1] = '.';
    padded[2] = '.';
    for (size_t i = 0; i < wordLen; i++) {
      padded[3 + i] = chars[i];
    }
    padded[wordLen + 3] = '.';
    padded[wordLen + 4] = '\0';

    size_t paddedLen = wordLen + 4;
    for (size_t i = 0; i + 3 < paddedLen; i++) {
      DatasetPair pair;
      pair.context[0] = Array_StoiGet(stoi, padded[i]);
      pair.context[1] = Array_StoiGet(stoi, padded[i + 1]);
      pair.context[2] = Array_StoiGet(stoi, padded[i + 2]);
      pair.target = Array_StoiGet(stoi, padded[i + 3]);
      Array_Append(dataset, &pair);
    }
  }

  return dataset;
}

Array *BuildTensorDataset(Context *ctx, Array *datasetPairs) {
  Array *tensorPairs = MakeArray(ctx->memory, sizeof(TensorPair), datasetPairs->size);

  for (size_t i = 0; i < datasetPairs->size; i++) {
    DatasetPair *dp = (DatasetPair *)Array_Idx(datasetPairs, i);

    Tensor *context = MakeFromContigousArray(ctx, SHAPE1D(3), dp->context, 3, I32);
    Tensor *target = MakeFromContigousArray(ctx, SHAPE1D(1), &dp->target, 1, I32);

    TensorPair tp = {.a = context, .b = target};
    Array_Append(tensorPairs, &tp);
  }

  return tensorPairs;
}

BatchedDataset BuildBatchedDataset(Context *ctx, Array *datasetPairs, size_t batchSize) {
  size_t numBatches = (datasetPairs->size + batchSize - 1) / batchSize;

  Array *batchInputs = MakeArray(ctx->memory, sizeof(Tensor *), numBatches);
  Array *batchTargets = MakeArray(ctx->memory, sizeof(Tensor *), numBatches);

  DatasetPair *pairs = (DatasetPair *)datasetPairs->items;

  for (size_t b = 0; b < numBatches; b++) {
    size_t start = b * batchSize;
    size_t end = start + batchSize;
    if (end > datasetPairs->size) {
      end = datasetPairs->size;
    }
    size_t currentBatchSize = end - start;

    i32 *inputData = allocate(ctx->memory, sizeof(i32) * currentBatchSize * 3);
    i32 *targetData = allocate(ctx->memory, sizeof(i32) * currentBatchSize);

    for (size_t i = 0; i < currentBatchSize; i++) {
      inputData[i * 3 + 0] = pairs[start + i].context[0];
      inputData[i * 3 + 1] = pairs[start + i].context[1];
      inputData[i * 3 + 2] = pairs[start + i].context[2];
      targetData[i] = pairs[start + i].target;
    }

    Tensor *inputTensor = MakeFromContigousArray(ctx, SHAPE2D(currentBatchSize, 3), inputData,
                                                 currentBatchSize * 3, I32);
    Tensor *targetTensor =
        MakeFromContigousArray(ctx, SHAPE1D(currentBatchSize), targetData, currentBatchSize, I32);

    Array_AppendTensor(batchInputs, inputTensor);
    Array_AppendTensor(batchTargets, targetTensor);
  }

  return (BatchedDataset){.inputs = batchInputs, .targets = batchTargets, .numBatches = numBatches};
}

typedef struct {
  FowardPassOp embedding;
  FowardPassOp l1;
  FowardPassOp l2;
  FowardPassOp l3;
  FowardPassOp bn1;
  FowardPassOp bn2;
  FowardPassOp tanh1;
  FowardPassOp tanh2;
  Optimizer optimizer;
  Dtype datatype;
} Model;

Model Make_Model(Context *ctx) {
  Model model;

  model.embedding = layer_Embedding(ctx, 27, 10);
  model.l1 = layer_Dense(ctx, 30, 100, false);
  model.l2 = layer_Dense(ctx, 100, 27, false);
  model.bn1 = layer_BatchNorm(ctx, 100);
  model.tanh1 = layer_Tanh(ctx);
  model.optimizer = optimizer_Adam(ctx, 0.001f);
  model.datatype = F32;

  return model;
}

Tensor *Model_Forward(Context *ctx, Model *model, Tensor *input, dim_t batchSize) {
  Tensor *out = Forward(ctx, &model->embedding, input);
  out = Reshape(ctx, out, SHAPE2D(batchSize, 30));

  out = Forward(ctx, &model->l1, out);
  out = Forward(ctx, &model->bn1, out);
  out = Forward(ctx, &model->tanh1, out);
  out = Forward(ctx, &model->l2, out);

  return out;
}

Array *Model_Parameters(Context *ctx, Model *model) {
  Array *params = MakeArray(ctx->memory, sizeof(Tensor *), 12);

  Array_AppendTensorArray(params, Parameters(ctx, &model->embedding));
  Array_AppendTensorArray(params, Parameters(ctx, &model->l1));
  Array_AppendTensorArray(params, Parameters(ctx, &model->l2));
  Array_AppendTensorArray(params, Parameters(ctx, &model->bn1));

  return params;
}

Array *Model_ParameterGradNorms(Context *ctx, Model *model) {
  Array *params = Model_Parameters(ctx, model);
  Array *gradNorms = MakeArray(ctx->memory, sizeof(Value), params->size);

  for (size_t i = 0; i < params->size; i++) {
    Tensor *p = Array_TensorIdx(params, i);
    Tensor *squared = Pow(ctx, p->grad, 2);
    Tensor *flat = Reshape(ctx, squared, SHAPE1D(p->grad->size));
    Tensor *totalSum = Sum(ctx, flat, 0);
    Tensor *norm = Sqrt(ctx, totalSum);

    Value normValue;
    VALUE_GET_FROM_ARR(norm->values, 0, &normValue, norm->dtype);
    Array_Append(gradNorms, &normValue);
  }

  return gradNorms;
}

static Tensor *softmax(Context *ctx, Tensor *logits, dim_t dim) {
  Tensor *maxVal = Max(ctx, logits, dim);
  Tensor *shifted = Subtract(ctx, logits, maxVal);
  Tensor *expVals = Exp(ctx, shifted);
  Tensor *sumExp = Sum(ctx, expVals, dim);
  return Divide(ctx, expVals, sumExp);
}

static int sampleFromProbs(Tensor *probs, dim_t numClasses) {
  f32 r = (f32)rand() / (f32)RAND_MAX;
  f32 cumulative = 0.0f;

  for (dim_t i = 0; i < numClasses; i++) {
    dim_t idx[2] = {0, i};
    Value *p = GetAt(probs, (Dim){.dims = idx, .numOfDims = 2});
    cumulative += p->as.f32;
    if (r <= cumulative) {
      return (int)i;
    }
  }

  return 0;
}

void Model_Generate(Context *ctx, Model *model, Array *itos, int numSamples, int maxNameLen,
                    dim_t vocabSize) {
  printf("\nGenerated names:\n");

  for (int sample = 0; sample < numSamples; sample++) {
    int context[3] = {0, 0, 0};
    char generated[256];
    int genLen = 0;

    for (int step = 0; step < maxNameLen; step++) {
      i32 inputData[3] = {context[0], context[1], context[2]};
      Tensor *input = MakeFromContigousArray(ctx, SHAPE2D(1, 3), inputData, 3, I32);

      Tensor *logits = Model_Forward(ctx, model, input, 1);
      Tensor *probs = softmax(ctx, logits, 1);

      int nextIdx = sampleFromProbs(probs, vocabSize);

      if (nextIdx == 0) {
        break;
      }

      generated[genLen++] = Array_ItosGet(itos, nextIdx);

      context[0] = context[1];
      context[1] = context[2];
      context[2] = nextIdx;
    }

    generated[genLen] = '\0';
    printf("%2d. %s\n", sample + 1, generated);
  }
}

void makemore_5() {
  Memory *mem = initializeArena((size_t)1024 * 1024 * 1024 * 5, 1); // 5GB
  Context ctx = {.memory = mem};

  printf("Arena capacity: %zu MB\n", mem->capacity / (1024 * 1024));

  Array *words = getNames(&ctx);
  printf("Got %zu words\n", words->size);

  Bitset *charset = MakeCharSet(&ctx, words);
  Array *stoi = MakeStoi(&ctx, charset);
  Array *itos = MakeItos(&ctx, charset);

  printf("Vocab size: %zu\n", itos->size);

  Array *dataset = BuildDataset(&ctx, words, stoi);
  printf("Total dataset pairs: %zu\n", dataset->size);

  BatchedDataset batchedData = BuildBatchedDataset(&ctx, dataset, BATCH_SIZE);
  printf("Total batches: %zu (batch_size=%d)\n", batchedData.numBatches, BATCH_SIZE);

  Model model = Make_Model(&ctx);
  Array *params = Model_Parameters(&ctx, &model);
  printf("Total parameters: %zu\n\n", params->size);

  size_t scratchBufferSize = (size_t)1024 * 1024 * 1024;
  void *scratchBuffer = allocate(mem, scratchBufferSize);
  Memory *scratchMem = initializeArenaWithBuffer(scratchBuffer, scratchBufferSize, 1);

  printf("Main arena after scratch alloc: %zu MB\n", mem->allocated / (1024 * 1024));
  printf("Scratch capacity: %zu MB\n", scratchMem->capacity / (1024 * 1024));

  ctx.isTraining = true;

  printf("Starting training with %zu samples in %zu batches\n", dataset->size,
         batchedData.numBatches);

  for (size_t epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    f32 totalLoss = 0.0f;
    size_t totalSamples = 0;
    f32 wsumAvg = 0;

    for (size_t b = 0; b < batchedData.numBatches; b++) {
      Tensor *input = *(Tensor **)Array_Idx(batchedData.inputs, b);
      Tensor *target = *(Tensor **)Array_Idx(batchedData.targets, b);

      dim_t batchSize = input->shape.dims[0];
      totalSamples += batchSize;

      Context scratchCtx = {.memory = scratchMem, .isTraining = true};

      Tensor *logits = Model_Forward(&scratchCtx, &model, input, batchSize);

      Tensor *targetOneHot = T_OneHot(&scratchCtx, target, 27);
      Tensor loss = loss_CrossEnthropy(&scratchCtx, targetOneHot, logits);

      Value lossValue;
      VALUE_GET_FROM_ARR(loss.values, 0, &lossValue, loss.dtype);
      totalLoss += lossValue.as.f32 * batchSize;

      Backward(&scratchCtx, &loss);

      if (epoch == 0 && b == 0) {
        printf("\nGradient norms after first batch:\n");
        Array *gradNorms = Model_ParameterGradNorms(&ctx, &model);
        for (size_t i = 0; i < gradNorms->size; i++) {
          Value *norm = Array_Idx(gradNorms, i);
          printf("  Param %zu grad norm: ", i);
          PRINT_VALUE(*norm);
          printf("\n");
        }
      }

      OptimizerStep(&ctx, &model.optimizer, params);
      ZeroGrad(&ctx, params);

      resetArena(scratchMem);
    }

    printf("Epoch %zu: Loss = %f\n", epoch, totalLoss / totalSamples);
  }

  printf("\nTraining complete. Generating samples...\n");

  Context genCtx = {.memory = mem, .isTraining = false};
  Model_Generate(&genCtx, &model, itos, 10, 20, (dim_t)itos->size);
}
