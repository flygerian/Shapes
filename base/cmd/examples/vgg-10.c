#include "common.h"
#include "nn/nn.h"
#include "raylib.h"
#include "result/result.h"
#include "shapes.h"
#include "tensor/types.h"
#include "tensor/value.h"
#include "utils_lib/array.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <glob.h>
#include <wctype.h>
#include <time.h>
#include "utils_lib/file.h"
#include "utils_lib/memory.h"
#include "utils_lib/utils_lib.h"
#include "utils_lib/error.h"
#include "string.h"
#include "examples.h"

#define DATASET_PATH     "base/cmd/examples/datasets/cifar-10-binary/cifar-10-batches-bin/data_batch_*.bin"
#define LABELS_FILE_PATH "base/cmd/examples/datasets/cifar-10-binary/cifar-10-batches-bin/batches.meta.txt"
#define TEST_BATCH_FILE  "base/cmd/examples/datasets/cifar-10-binary/cifar-10-batches-bin/test_batch.bin"
#define IMAGES_WIDTH     32
#define IMAGES_HEIGHT    32
#define NUM_CHANNELS     3
#define CHANNEL_PLANE    IMAGES_WIDTH *IMAGES_HEIGHT
#define BATCH_SIZE       64
#define NUM_EPOCHS       1

typedef struct dataset {
  Array_Tensor Xtrain;
  Array_Tensor Ytrain;
  Array_Tensor Xtest;
  Array_Tensor Ytest;
  Array_Tensor labels;
} dataset;

typedef struct {
  FowardPassOp *convBlock;
  FowardPassOp *linearBlock;
} Vgg10Model;

int globErrFn(const char *epath, int eerrno) {
  printf("Err in path %s errno %d", epath, eerrno);
}

static inline Error readNextImageChannelsAndLabel(File imageFile, byte *out_label, byte *out_r, byte *out_g, byte *out_b) {
  Error readErr = File_ReadBytesToBuffer(imageFile, out_label, 1);
  RETURN_ON_ERROR(readErr);

  // Read the R channel
  readErr = File_ReadBytesToBuffer(imageFile, out_r, CHANNEL_PLANE);
  RETURN_ON_ERROR(readErr);

  // Read the B channel
  readErr = File_ReadBytesToBuffer(imageFile, out_g, CHANNEL_PLANE);
  RETURN_ON_ERROR(readErr);
  // Read the C channel
  readErr = File_ReadBytesToBuffer(imageFile, out_b, CHANNEL_PLANE);
  RETURN_ON_ERROR(readErr);

  return OK;
}

static inline void readAsNCHWIntoArray(Array *image, byte *restrict r, byte *restrict g, byte *restrict b) {
  for (RANGE(pixelIdx, CHANNEL_PLANE)) {
    f32 pixel[3] = {F32_(r[pixelIdx]) / 255, F32_(g[pixelIdx]) / 255, F32_(b[pixelIdx]) / 255};
    Array_AppendF32Buffer(image, pixel, 3);
  }
}

dataset getDataset(Context *ctx) {
  glob_t batchFiles;
  i32 err = glob(DATASET_PATH, GLOB_ERR, globErrFn, &batchFiles);

  PANIC_WITH_MSG_IF(err != 0, "could error while globbing dataset path");

  File labelsDataFile = File_OpenPathInReadMode(LABELS_FILE_PATH);
  Array *labels = File_ReadLines(ctx->memory, labelsDataFile);
  File testBatchFile = File_OpenPathInReadMode(TEST_BATCH_FILE);

  for (RANGE(i, labels->size)) {
    string label = Array_Idx(labels, i);
    printf("\n%s\n", label);
  }

  Array_Tensor Xtrain = Make_DynamicTensorArray(ctx->memory);
  Array_Tensor Ytrain = Make_DynamicTensorArray(ctx->memory);
  Array_Tensor Xtest = Make_DynamicTensorArray(ctx->memory);
  Array_Tensor Ytest = Make_DynamicTensorArray(ctx->memory);

  byte label[1] = {0};
  byte r[CHANNEL_PLANE] = {0};
  byte g[CHANNEL_PLANE] = {0};
  byte b[CHANNEL_PLANE] = {0};

  Array_F32 image = Make_F32Array(ctx->memory, CHANNEL_PLANE * NUM_CHANNELS);

  size_t numImagesProcessed = 0;
  for (RANGE(i, batchFiles.gl_pathc)) {
    string path = batchFiles.gl_pathv[i];

    fprintf(stdout, "\nProcessing Path: %s\n", path);

    File bf = File_OpenPathInReadMode(path);
    Error readError;

    while (true) {
      Error err = readNextImageChannelsAndLabel(bf, label, r, g, b);
      if (err.code != 0) {
        break;
      }

      readAsNCHWIntoArray(image, r, g, b);

      Tensor *imageTensor = MakeFromContigousArray(ctx, SHAPE3D(IMAGES_HEIGHT, IMAGES_WIDTH, NUM_CHANNELS), image->items, F32);
      Array_AppendTensor(Xtrain, imageTensor);

      f32 lbl[1] = {F32_(label[0])};
      Tensor *labelTensor = MakeFromContigousArray(ctx, SCALAR, lbl, F32);
      Array_AppendTensor(Ytrain, labelTensor);

      Array_Reset(image);
      numImagesProcessed += 1;

      // printf("\nProcessed %zu images", numImagesProcessed);
    }

    CloseFile(&bf);
  }

  while (true) {
    Error err = readNextImageChannelsAndLabel(testBatchFile, label, r, g, b);
    if (err.code != 0) {
      break;
    }

    readAsNCHWIntoArray(image, r, g, b);
    Tensor *imageTensor = MakeFromContigousArray(ctx, SHAPE3D(IMAGES_HEIGHT, IMAGES_WIDTH, NUM_CHANNELS), image->items, F32);
    Array_AppendTensor(Xtest, imageTensor);

    f32 lbl[1] = {F32_(label[0])};
    Tensor *labelTensor = MakeFromContigousArray(ctx, SCALAR, lbl, F32);
    Array_AppendTensor(Ytest, labelTensor);
    Array_Reset(image);
  }

  CloseFile(&labelsDataFile);
  CloseFile(&testBatchFile);
  return (dataset){.Xtrain = Xtrain, .Ytrain = Ytrain, .Xtest = Xtest, .Ytest = Ytest, .labels = labels};
}

static inline ArrayPair toBatches(Context *ctx, Array *X, Array *Y) {
  size_t amountProcessed = 0;
  Array_Tensor currentXBatch = Make_TensorArray(ctx->memory, BATCH_SIZE);
  Array_Tensor currentYBatch = Make_TensorArray(ctx->memory, BATCH_SIZE);

  Array *Xbatched = Make_DynamicTensorArray(ctx->memory);
  Array *Ybatched = Make_DynamicTensorArray(ctx->memory);

  u8 counter = 0;
  while (true) {
    bool isAtEnd = (amountProcessed + counter) >= X->size;
    if (counter == BATCH_SIZE || isAtEnd) {
      Tensor *batchedXTensor = Stack(ctx, currentXBatch);
      Tensor *batchedYTensor = Stack(ctx, currentYBatch);

      Array_AppendTensor(Xbatched, batchedXTensor);
      Array_AppendTensor(Ybatched, batchedYTensor);

      counter = 0;
      Array_Reset(currentXBatch);
      Array_Reset(currentYBatch);

      amountProcessed = Xbatched->size * BATCH_SIZE;

      if (isAtEnd) {
        break;
      }
    }

    Tensor *tXTrain = Array_TensorIdx(X, amountProcessed + counter);
    Array_AppendTensor(currentXBatch, tXTrain);

    Tensor *tYTrain = Array_TensorIdx(Y, amountProcessed + counter);
    Array_AppendTensor(currentYBatch, tYTrain);

    counter += 1;
  }

  return ARRAY_PAIR(Xbatched, Ybatched);
}

FowardPassOp *Make_ConvBlock(Context *ctx) {
  FowardPassOp *layers[] = {layer_Conv2d(ctx, F32, 3, 64, 2, 2, 1, false),
                            layer_Relu(ctx, F32),
                            layer_Conv2d(ctx, F32, 64, 64, 2, 2, 1, false),
                            layer_Relu(ctx, F32),

                            layer_MaxPool2d(ctx, F32, 2, 2, 1),

                            layer_Conv2d(ctx, F32, 64, 128, 2, 2, 1, false),
                            layer_Relu(ctx, F32),
                            layer_Conv2d(ctx, F32, 128, 128, 2, 2, 1, false),
                            layer_Relu(ctx, F32),

                            layer_MaxPool2d(ctx, F32, 2, 2, 1),

                            layer_Conv2d(ctx, F32, 128, 256, 2, 2, 1, false),
                            layer_Relu(ctx, F32),
                            layer_Conv2d(ctx, F32, 256, 256, 2, 2, 1, false),
                            layer_Relu(ctx, F32),

                            layer_MaxPool2d(ctx, F32, 2, 2, 1),

                            layer_Conv2d(ctx, F32, 256, 512, 2, 2, 1, false),
                            layer_Relu(ctx, F32),
                            layer_AdaptiveAvgPool2d(ctx, F32, 1, 1)};

  return Make_Sequential(ctx, layers, 18, F32);
}

FowardPassOp *Make_LinearBlock(Context *ctx, u8 numLabels) {
  FowardPassOp *layers[] = {layer_Flatten(ctx, F32), layer_Dense(ctx, F32, 512, 4000, false),
                            layer_Relu(ctx, F32),    layer_Dense(ctx, F32, 4000, 1000, false),
                            layer_Relu(ctx, F32),    layer_Dense(ctx, F32, 1000, numLabels, false)};

  return Make_Sequential(ctx, layers, 6, F32);
}

FowardPassOp *Make_Model(Context *ctx, u8 numLabels) {
  FowardPassOp *convBlock = Make_ConvBlock(ctx);
  FowardPassOp *linearBlock = Make_LinearBlock(ctx, numLabels);

  FowardPassOp *blocks[2] = {convBlock, linearBlock};
  return Make_Sequential(ctx, blocks, 2, F32);
}

FowardPassOp *runTraining(Context *hostCtx, dataset ds) {
  // Tensor *first = Array_TensorIdx(ds.Xtest, 246);
  // basicRaylibWindow(first);

  ArrayPair trainDs = toBatches(hostCtx, ds.Xtrain, ds.Ytrain);
  Array *Xtrain = trainDs.a;
  Array *Ytrain = trainDs.b;

  Context cudaCtx = InitializeCudaContext(10 * GB);
  MoveToCuda(&cudaCtx, Xtrain);
  MoveToCuda(&cudaCtx, Ytrain);

  printf("\n\n%zu batches created from dataset %zu size\n", Xtrain->size, ds.Xtrain->size);

  u8 numLabels = ds.labels->size;

  FowardPassOp *model = Make_Model(&cudaCtx, numLabels);
  Optimizer *optimzer = optimizer_SGD(&cudaCtx, 0.0001);

  Context scratch = cudaCtx; //GetScratchContext(hostCtx, 10 * GB);

  for (RANGE(e, NUM_EPOCHS)) {
    size_t totalEpochLoss = 0;
    f32 totalLoss = 0.0f;
    size_t totalSamples = 0;
    f32 wsumAvg = 0;

    clock_t epochStart = clock();
    struct timespec epochWallStart;
    clock_gettime(CLOCK_MONOTONIC, &epochWallStart);
    for (RANGE(i, Xtrain->size)) {

      clock_t batchStart = clock();
      struct timespec wallStart;
      clock_gettime(CLOCK_MONOTONIC, &wallStart);

      if (i % 100 == 0) {
        printf("Processing batch %zu of %zu \n", i, Xtrain->size - 1);
      }

      Tensor *batch = Array_TensorIdx(Xtrain, i);
      Tensor *logits = Forward(&scratch, model, batch);

      Tensor *ybatch = Array_TensorIdx(Ytrain, i);
      Tensor *yOneHot = Squeeze(hostCtx, T_OneHot(&scratch, ybatch, numLabels));
      Tensor loss = loss_CrossEnthropy(&scratch, yOneHot, logits);
      Flush(&scratch);

      Value *lossValue = GetAt(&loss, SHAPE1D(0));
      totalLoss += lossValue->as.f32;
      totalSamples += 1;

      Backward(&scratch, &loss);
      Array *parameters = Parameters(&scratch, model);
      OptimizerStep(&scratch, optimzer, parameters);
      ZeroGrad(&scratch, parameters);
      resetArena(scratch.memory);

      double batchTime = (double)(clock() - batchStart) / CLOCKS_PER_SEC;
      struct timespec wallEnd;
      clock_gettime(CLOCK_MONOTONIC, &wallEnd);
      double wallTime = (wallEnd.tv_sec - wallStart.tv_sec) + (wallEnd.tv_nsec - wallStart.tv_nsec) / 1e9;
      printf("batch %zu, CPU Time = %.3fs, Wall Time = %.3fs\n", i, batchTime, wallTime);
    }

    double epochTime = (double)(clock() - epochStart) / CLOCKS_PER_SEC;
    struct timespec epochWallEnd;
    clock_gettime(CLOCK_MONOTONIC, &epochWallEnd);
    double epochWallTime = (epochWallEnd.tv_sec - epochWallStart.tv_sec) + (epochWallEnd.tv_nsec - epochWallStart.tv_nsec) / 1e9;
    printf("Epoch %zu: Avg batch Loss = %f, CPU Time = %.3fs, Wall Time = %.3fs\n", e, totalLoss / totalSamples, epochTime, epochWallTime);
  }

  popScratch(scratch.memory);

  return model;
}

void runInference(Context *ctx, FowardPassOp *model, dataset ds) {
  ArrayPair testDs = toBatches(ctx, ds.Xtest, ds.Ytest);
  Array *Xtest = testDs.a;
  Array *Ytest = testDs.b;

  Context scratch = GetScratchContext(ctx, 10 * GB);

  for (RANGE(i, BATCH_SIZE)) {
    Tensor *batch = Array_TensorIdx(Xtest, i);
    Tensor *logits = Forward(&scratch, model, batch);

    u8 oneInput = {1};
    Tensor *ybatch = Array_TensorIdx(Ytest, i);

    // printf("\n Ybatch: \n");
    // PrintTensor(ybatch);
    // printf("\n \n");

    Tensor *logitsProbs = nn_Softmax(&scratch, logits);
    
    // printf("\n logitsProbs: \n");
    // PrintTensor(logitsProbs);
    // printf("\n\n");

    Tensor *predictions = ArgMax(&scratch, logitsProbs, logitsProbs->shape.numOfDims - 1);
    PANIC_IF(predictions->shape.numOfDims != ybatch->shape.numOfDims, ERR_DIM_MISMATCH);

    printf("\n predictions: \n");
    PrintTensor(predictions);
    printf("\n\n");

    Tensor *compMask = Equal(ctx, Cast(&scratch, predictions, F32), ybatch);

    printf("\n compmask: \n");
    PrintTensor(compMask);
    printf("\n\n");

    bool *values = compMask->values;
    size_t ones = 0;
    for (RANGE(iv, compMask->size)) {
      bool matched = values[iv];
      if (matched) {
        ones += 1;
      }
    }

    printf("\n Batch Result %zu of %zu\n", ones, compMask->size);
    printf("===============================================================================");
    resetArena(scratch.memory);
  }
}

void vgg10() {
  Context hostCtx = InitializeHostContext(25 * GB, 1);

  dataset ds = getDataset(&hostCtx);

  FowardPassOp *model = runTraining(&hostCtx, ds);
  runInference(&hostCtx, model, ds);
}
