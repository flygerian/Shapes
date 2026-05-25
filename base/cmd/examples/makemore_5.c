#include "common.h"
#include "nn/nn.h"
#include "nn/nn_internal.h"
#include "result/result.h"
#include "shapes.h"
#include "types.h"
#include "utils_lib/utils_lib.h"
#include "value.h"
#include "utils_lib/array.h"
#include "utils_lib/bitset.h"
#include "utils_lib/memory.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BATCH_SIZE 128
#define NUM_EPOCHS 100

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

    Tensor *context = shapes_Make_FromContigousArray(ctx, SHAPE1D(3), dp->context, I32);
    Tensor *target = shapes_Make_FromContigousArray(ctx, SHAPE1D(1), &dp->target, I32);

    TensorPair tp = {.a = Cast(ctx, context, F32), .b = Cast(ctx, target, F32)};
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

    Tensor *inputTensor = shapes_Make_FromContigousArray(ctx, SHAPE2D(currentBatchSize, 3), inputData, I32);
    Tensor *targetTensor = shapes_Make_FromContigousArray(ctx, SHAPE1D(currentBatchSize), targetData, I32);

    shapes_Array_AppendTensor(batchInputs, inputTensor);
    shapes_Array_AppendTensor(batchTargets, targetTensor);
  }

  return (BatchedDataset){.inputs = batchInputs, .targets = batchTargets, .numBatches = numBatches};
}

typedef struct {
  FowardPassOp *embedding;
  FowardPassOp *layers;
  Optimizer *optimizer;
  Dtype datatype;
} Model;

Model Make_Model(Context *ctx) {
  Model model;

  model.embedding = shapesnn_Embedding(ctx, F32, 27, 10);

  FowardPassOp *layers[] = {
      shapesnn_Dense(ctx, F32, 30, 100, false),
      shapesnn_BatchNorm(ctx, F32, 100),
      shapesnn_Tanh(ctx, F32),
      shapesnn_Dense(ctx, F32, 100, 27, false),
  };

  model.layers = shapesnn_Sequential(ctx, layers, 4, F32);
  model.optimizer = shapesnn_Adam(ctx, 0.01f);

  return model;
}

Tensor *Model_Forward(Context *ctx, Model *model, Tensor *input, dim_t batchSize) {
  return sequentialModelForward(ctx, model->layers, input);
}

Array *Model_Parameters(Context *ctx, Model *model) {
  return sequentialModelParameters(ctx, model->layers);
}

Array *Model_ParameterGradNorms(Context *ctx, Model *model) {
  Array *params = Model_Parameters(ctx, model);
  Array *gradNorms = MakeArray(ctx->memory, sizeof(Value), params->size);

  for (size_t i = 0; i < params->size; i++) {
    Tensor *p = shapes_Array_TensorIdx(params, i);
    Tensor *squared = shapes_Pow(ctx, p->grad, 2);
    Tensor *flat = shapes_Reshape(ctx, squared, SHAPE1D(p->grad->size));
    Tensor *totalSum = shapes_Sum(ctx, flat, 0);
    Tensor *norm = shapes_Sqrt(ctx, totalSum);

    Value normValue;
    VALUE_GET_FROM_ARR(norm->values, 0, &normValue, norm->dtype);
    Array_Append(gradNorms, &normValue);
  }

  return gradNorms;
}

static Tensor *softmax(Context *ctx, Tensor *logits, dim_t dim) {
  Tensor *maxVal = shapes_Max(ctx, logits, dim);
  Tensor *shifted = shapes_Subtract(ctx, logits, maxVal);
  Tensor *expVals = shapes_Exp(ctx, shifted);
  Tensor *sumExp = shapes_Sum(ctx, expVals, dim);
  return shapes_Divide(ctx, expVals, sumExp);
}

static int sampleFromProbs(Tensor *probs, dim_t numClasses) {
  f32 r = (f32)rand() / (f32)RAND_MAX;
  f32 cumulative = 0.0f;

  for (dim_t i = 0; i < numClasses; i++) {
    dim_t idx[2] = {0, i};
    Value *p = shapes_GetAt(probs, (Dim){.dims = idx, .numOfDims = 2});
    cumulative += p->as.f32;
    if (r <= cumulative) {
      return (int)i;
    }
  }

  return 0;
}

void Model_Generate(Context *ctx, Model *model, Array *itos, int numSamples, int maxNameLen, dim_t vocabSize) {
  printf("\nGenerated names:\n");

  for (int sample = 0; sample < numSamples; sample++) {
    int context[3] = {0, 0, 0};
    char generated[256];
    int genLen = 0;

    for (int step = 0; step < maxNameLen; step++) {
      i32 inputData[3] = {context[0], context[1], context[2]};
      Tensor *input = shapes_Make_FromContigousArray(ctx, SHAPE2D(1, 3), inputData, I32);

      Tensor *embeddings = shapesnn_Forward(ctx, model->embedding, input);
      Tensor *reshapedEmbeddings = shapes_Reshape(ctx, embeddings, SHAPE2D(1, 30));
      Tensor *logits = Model_Forward(ctx, model, reshapedEmbeddings, 1);
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
  Context ctx = shapes_InitializeHostContext(5 * GB, 1);

  printf("Arena capacity: %zu MB\n", ctx.memory->capacity / MB);

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

  size_t scratchBufferSize = (size_t)1024 * 1024 * 10;

  ctx.isTraining = true;

  printf("Starting training with %zu samples in %zu batches\n", dataset->size, batchedData.numBatches);

  Context scratchCtx = shapes_GetScratchContext(&ctx, scratchBufferSize);
  printf("Scratch capacity: %zu MB\n", scratchCtx.memory->capacity / (1024 * 1024));

  for (size_t epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    f32 totalLoss = 0.0f;
    size_t totalSamples = 0;
    f32 wsumAvg = 0;

    clock_t epochStart = clock();

    for (size_t b = 0; b < batchedData.numBatches; b++) {
      Tensor *input = *(Tensor **)Array_Idx(batchedData.inputs, b);
      Tensor *target = *(Tensor **)Array_Idx(batchedData.targets, b);

      dim_t batchSize = input->shape.dims[0];
      totalSamples += batchSize;


      Tensor *embeddings = shapesnn_Forward(&scratchCtx, model.embedding, input);
      Tensor *reshapedEmbeddings = shapes_Reshape(&scratchCtx, embeddings, SHAPE2D(batchSize, 30));
      Tensor *logits = Model_Forward(&scratchCtx, &model, reshapedEmbeddings, batchSize);

      Tensor *targetOneHot = shapes_Make_OneHotTensor(&scratchCtx, target, 27);
      Tensor loss = shapesnn_CrossEnthropy(&scratchCtx, targetOneHot, logits);

      Value lossValue;
      VALUE_GET_FROM_ARR(loss.values, 0, &lossValue, loss.dtype);
      totalLoss += lossValue.as.f32 * batchSize;

      shapesnn_Backward(&scratchCtx, &loss);

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

      if (epoch == 0 && b == 0) {
        printf("Scratch used per batch: %zu KB\n", scratchCtx.memory->allocated / 1024);
      }

      shapesnn_OptimizerStep(&ctx, model.optimizer, params);
      shapesnn_ZeroGrad(&ctx, params);

      if (epoch == 0 && b == 0) {
        printf("Scratch used per batch: %zu KB\n", scratchCtx.memory->allocated / 1024);
      }

      resetArena(scratchCtx.memory);
    }

    double epochTime = (double)(clock() - epochStart) / CLOCKS_PER_SEC;
    printf("Epoch %zu: Loss = %f, Time = %.3fs\n", epoch, totalLoss / totalSamples, epochTime);
  }

  printf("\nTraining complete. Generating samples...\n");

  Model_Generate(&scratchCtx, &model, itos, 10, 20, (dim_t)itos->size);
}
