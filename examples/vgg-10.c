#include "nn.h"
#include "result.h"
#include "shapes.h"
#include "types.h"
#include "array.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <glob.h>
#include <time.h>
#include "shapescuda.h"
#include "file.h"
#include "memory.h"
#include "error.h"
#include "string.h"

#define DATASET_PATH     "cifar-10-batches-bin/data_batch_*.bin"
#define LABELS_FILE_PATH "cifar-10-batches-bin/batches.meta.txt"
#define TEST_BATCH_FILE  "cifar-10-batches-bin/test_batch.bin"
#define IMAGES_WIDTH     32
#define IMAGES_HEIGHT    32
#define NUM_CHANNELS     3
#define CHANNEL_PLANE    IMAGES_WIDTH * IMAGES_HEIGHT
#define BATCH_SIZE       16
#define NUM_EPOCHS       40
#define LEARNING_RATE    0.001

typedef struct dataset {
  shapes_ArrayTensor Xtrain;
  shapes_ArrayTensor Ytrain;
  shapes_ArrayTensor Xtest;
  shapes_ArrayTensor Ytest;
  shapes_ArrayTensor labels;
} dataset;

typedef struct {
  shapesnn_FowardPassOp *convBlock;
  shapesnn_FowardPassOp *linearBlock;
} Vgg10Model;

int globErrFn(const char *epath, int eerrno) {
  printf("Err in path %s errno %d", epath, eerrno);
  return 0;
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

static inline void readAsNCHWIntoArray(olib_Array *image, byte *restrict r, byte *restrict g, byte *restrict b) {
  for (RANGE(pixelIdx, CHANNEL_PLANE)) {
    f32 pixel[3] = {F32_(r[pixelIdx]) / 255, F32_(g[pixelIdx]) / 255, F32_(b[pixelIdx]) / 255};
    olib_ArrayAppendF32Buffer(image, pixel, 3);
  }
}

dataset getDataset(shapes_Context *ctx) {
  glob_t batchFiles;
  i32 err = glob(DATASET_PATH, GLOB_ERR, globErrFn, &batchFiles);

  PANIC_WITH_MSG_IF(err != 0, "could error while globbing dataset path");

  File labelsDataFile = File_OpenPathInReadMode(LABELS_FILE_PATH);
  olib_Array *labels = File_ReadLines(ctx->memory, labelsDataFile);
  File testBatchFile = File_OpenPathInReadMode(TEST_BATCH_FILE);

  for (RANGE(i, labels->size)) {
    string label = olib_ArrayIdx(labels, i);
    printf("\n%s\n", label);
  }

  shapes_ArrayTensor Xtrain = shapes_MakeTensorArray(ctx->memory, 50000);
  shapes_ArrayTensor Ytrain = shapes_MakeTensorArray(ctx->memory, 50000);
  shapes_ArrayTensor Xtest = shapes_MakeTensorArray(ctx->memory, 10000);
  shapes_ArrayTensor Ytest = shapes_MakeTensorArray(ctx->memory, 10000);

  byte label[1] = {0};
  byte r[CHANNEL_PLANE] = {0};
  byte g[CHANNEL_PLANE] = {0};
  byte b[CHANNEL_PLANE] = {0};

  olib_ArrayF32 image = olib_MakeF32Array(ctx->memory, (size_t)CHANNEL_PLANE * NUM_CHANNELS);

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

      Tensor imageTensor = shapes_MakeFromContigousArray(ctx, SHAPE3D(IMAGES_HEIGHT, IMAGES_WIDTH, NUM_CHANNELS), image->items, F32);
      shapes_ArrayAppendTensor(Xtrain, &imageTensor);

      f32 lbl[1] = {F32_(label[0])};
      Tensor labelTensor = shapes_MakeFromContigousArray(ctx, SCALAR, lbl, F32);
      shapes_ArrayAppendTensor(Ytrain, &labelTensor);

      olib_ArrayReset(image);
      numImagesProcessed += 1;
    }

    CloseFile(&bf);
  }

  while (true) {
    Error err = readNextImageChannelsAndLabel(testBatchFile, label, r, g, b);
    if (err.code != 0) {
      break;
    }

    readAsNCHWIntoArray(image, r, g, b);
    Tensor imageTensor = shapes_MakeFromContigousArray(ctx, SHAPE3D(IMAGES_HEIGHT, IMAGES_WIDTH, NUM_CHANNELS), image->items, F32);
    shapes_ArrayAppendTensor(Xtest, &imageTensor);

    f32 lbl[1] = {F32_(label[0])};
    Tensor labelTensor = shapes_MakeFromContigousArray(ctx, SCALAR, lbl, F32);
    shapes_ArrayAppendTensor(Ytest, &labelTensor);
    olib_ArrayReset(image);
  }

  CloseFile(&labelsDataFile);
  CloseFile(&testBatchFile);
  return (dataset){.Xtrain = Xtrain, .Ytrain = Ytrain, .Xtest = Xtest, .Ytest = Ytest, .labels = labels};
}

static inline olib_ArrayPair toBatches(shapes_Context *ctx, olib_Array *X, olib_Array *Y) {
  size_t amountProcessed = 0;
  shapes_ArrayTensor currentXBatch = shapes_MakeTensorArray(ctx->memory, BATCH_SIZE);
  shapes_ArrayTensor currentYBatch = shapes_MakeTensorArray(ctx->memory, BATCH_SIZE);

  olib_Array *Xbatched = shapes_MakeTensorArray(ctx->memory, 4000);
  olib_Array *Ybatched = shapes_MakeTensorArray(ctx->memory, 4000);

  u8 counter = 0;
  while (true) {
    bool isAtEnd = (amountProcessed + counter) >= X->size;
    if (counter == BATCH_SIZE || isAtEnd) {
      Tensor batchedXTensor = shapes_Stack(ctx, currentXBatch);
      Tensor batchedYTensor = shapes_Stack(ctx, currentYBatch);

      shapes_ArrayAppendTensor(Xbatched, &batchedXTensor);
      shapes_ArrayAppendTensor(Ybatched, &batchedYTensor);

      counter = 0;
      olib_ArrayReset(currentXBatch);
      olib_ArrayReset(currentYBatch);

      amountProcessed = Xbatched->size * BATCH_SIZE;

      if (isAtEnd) {
        break;
      }
    }

    Tensor tXTrain = shapes_ArrayTensorIdx(X, amountProcessed + counter);
    shapes_ArrayAppendTensor(currentXBatch, &tXTrain);

    Tensor tYTrain = shapes_ArrayTensorIdx(Y, amountProcessed + counter);
    shapes_ArrayAppendTensor(currentYBatch, &tYTrain);

    counter += 1;
  }

  return ARRAY_PAIR(Xbatched, Ybatched);
}

shapesnn_FowardPassOp Make_ConvBlock(shapes_Context *ctx) {
  shapesnn_FowardPassOp layers[] = {
    shapesnn_Conv2d(ctx, F32, 3, 64, 2, 2, 1, false),
    shapesnn_Relu(ctx, F32),
    shapesnn_Conv2d(ctx, F32, 64, 64, 2, 2, 1, false),
    shapesnn_Relu(ctx, F32),

    shapesnn_MaxPool2d(ctx, F32, 2, 2, 1),

    shapesnn_Conv2d(ctx, F32, 64, 128, 2, 2, 1, false),
    shapesnn_Relu(ctx, F32),
    shapesnn_Conv2d(ctx, F32, 128, 128, 2, 2, 1, false),
    shapesnn_Relu(ctx, F32),

    shapesnn_MaxPool2d(ctx, F32, 2, 2, 1),

    shapesnn_Conv2d(ctx, F32, 128, 256, 2, 2, 1, false),
    shapesnn_Relu(ctx, F32),
    shapesnn_Conv2d(ctx, F32, 256, 256, 2, 2, 1, false),
    shapesnn_Relu(ctx, F32),

    shapesnn_MaxPool2d(ctx, F32, 2, 2, 1),

    shapesnn_Conv2d(ctx, F32, 256, 512, 2, 2, 1, false),
    shapesnn_Relu(ctx, F32),
    shapesnn_AdaptiveAvgPool2d(ctx, F32, 1, 1)
  };

  return shapesnn_Sequential(ctx, layers, 18, F32);
}

shapesnn_FowardPassOp Make_LinearBlock(shapes_Context *ctx, u8 numLabels) {
  shapesnn_FowardPassOp layers[] = {
    shapesnn_Flatten(ctx, F32), 
    shapesnn_Dense(ctx, F32, 512, 4000, false),      
    shapesnn_Relu(ctx, F32), 
    shapesnn_Dense(ctx, F32, 4000, 1000, false),
    shapesnn_Relu(ctx, F32),    
    shapesnn_Dense(ctx, F32, 1000, numLabels, false)
  };

  return shapesnn_Sequential(ctx, layers, 6, F32);
}

shapesnn_FowardPassOp Make_Model(shapes_Context *ctx, u8 numLabels) {
  shapesnn_FowardPassOp convBlock = Make_ConvBlock(ctx);
  shapesnn_FowardPassOp linearBlock = Make_LinearBlock(ctx, numLabels);

  shapesnn_FowardPassOp blocks[2] = {convBlock, linearBlock};
  return shapesnn_Sequential(ctx, blocks, 2, F32);
}

shapesnn_FowardPassOp runTraining(shapes_Context *hostCtx, shapes_Context *cudaCtx, dataset ds) {
  u8 numLabels = ds.labels->size;
  shapesnn_FowardPassOp model = Make_Model(cudaCtx, numLabels);
  shapesnn_Optimizer optimzer = shapesnn_SGD(cudaCtx, LEARNING_RATE);

  olib_ArrayPair trainDs = toBatches(hostCtx, ds.Xtrain, ds.Ytrain);
  olib_Array *Xtrain = trainDs.a;
  olib_Array *Ytrain = trainDs.b;

  shapes_Context dsCudaCtx = shapes_GetScratchContext(cudaCtx, 1 * GB);

  shapes_MoveToCuda(&dsCudaCtx, Xtrain);
  shapes_MoveToCuda(&dsCudaCtx, Ytrain);

  printf("\n\n%zu batches created from dataset %zu size\n", Xtrain->size, ds.Xtrain->size);

  shapes_Context scratch = shapes_GetScratchContext(cudaCtx, 4 * GB);

  for (RANGE(e, NUM_EPOCHS)) {
    size_t totalEpochLoss = 0;
    f32 totalLoss = 0.0f;
    size_t totalSamples = 0;
    f32 wsumAvg = 0;

    clock_t epochStart = clock();
    struct timespec epochWallStart;
    clock_gettime(CLOCK_MONOTONIC, &epochWallStart);
    for (RANGE(i, Xtrain->size)) {
      struct timespec wallStart;
      clock_gettime(CLOCK_MONOTONIC, &wallStart);

      if (i % 100 == 0) {
        printf("Processing batch %zu of %zu \n", i, Xtrain->size - 1);
      }

      Tensor batch = shapes_ArrayTensorIdx(Xtrain, i);
      Tensor logits = shapesnn_Forward(&scratch, &model, &batch);

      Tensor ybatch = shapes_ArrayTensorIdx(Ytrain, i);
      Tensor oneHotPtr = shapes_MakeOneHotTensor(&scratch, &ybatch, numLabels);
      Tensor yOneHotVal = shapes_Squeeze(hostCtx, &oneHotPtr);
      Tensor *yOneHot = &yOneHotVal;
      Tensor loss = shapesnn_CrossEnthropy(&scratch, yOneHot, &logits);

      shapes_Value *lossValue = shapes_GetAt(&loss, SHAPE1D(0));
      totalLoss += lossValue->as.f32;
      totalSamples += 1;

      shapesnn_Backward(&scratch, &loss);
      olib_Array *parameters = shapesnn_Parameters(&scratch, &model);
      shapesnn_OptimizerStep(&scratch, &optimzer, parameters);
      shapesnn_ZeroGrad(&scratch, parameters);

      olib_ResetArena(scratch.memory);
      shapescuda_RewindMemory(&scratch.cudaMemory);
    }

    double epochTime = (double)(clock() - epochStart) / CLOCKS_PER_SEC;
    struct timespec epochWallEnd;
    clock_gettime(CLOCK_MONOTONIC, &epochWallEnd);
    double epochWallTime = (epochWallEnd.tv_sec - epochWallStart.tv_sec) + (epochWallEnd.tv_nsec - epochWallStart.tv_nsec) / 1e9;
    printf("Epoch %zu: Avg batch Loss = %f, CPU Time = %.3fs, Wall Time = %.3fs\n", e, totalLoss / totalSamples, epochTime, epochWallTime);
  }

  shapes_ArrayNamedTensor modelTensors = shapesnn_Tensors(&scratch, &model);
  for(RANGE(i, modelTensors->size)) {
    shapesnn_NamedTensor t = shapes_Array_NamedTensorIdx(modelTensors, i);
    shapes_MoveTensorToHost(hostCtx, &t.tensor);
    olib_ArraySetAt(modelTensors, i, &t);
  }

  shapesnn_SafeTensors_Save(&scratch, modelTensors, "vgg10.safetensors");
  shapescuda_FreeScratchMemory(&dsCudaCtx.cudaMemory);
  // FreeCudaScratchMemory(&scratch.cudaMemory);

  return model;
}

void runInference(shapes_Context *hostCtx, shapesnn_FowardPassOp *model, dataset ds) {
  olib_ArrayPair testDs = toBatches(hostCtx, ds.Xtest, ds.Ytest);
  olib_Array *Xtest = testDs.a;
  olib_Array *Ytest = testDs.b;

  shapes_Context cudaCtx = shapes_InitializeCudaContext(6 * GB);
  shapes_MoveToCuda(&cudaCtx, Xtest);
  shapes_MoveToCuda(&cudaCtx, Ytest);

  shapes_Context scratch = shapes_GetScratchContext(&cudaCtx, 5 * GB);
  for (RANGE(i, BATCH_SIZE)) {
    Tensor batch = shapes_ArrayTensorIdx(Xtest, i);
    Tensor logits = shapesnn_Forward(&scratch, model, &batch);

    u8 oneInput = {1};
    Tensor ybatch = shapes_ArrayTensorIdx(Ytest, i);

    Tensor logitsProbs = shapesnn_Softmax(&scratch, &logits);

    Tensor predictionsVal = shapes_ArgMax(&scratch, &logitsProbs, logitsProbs.shape.numOfDims - 1);
    Tensor *predictions = &predictionsVal;
    PANIC_IF(predictions->shape.numOfDims != ybatch.shape.numOfDims, ERR_DIM_MISMATCH);

    Tensor castedPred = Cast(&scratch, predictions, F32);
    Tensor compMaskVal = shapes_Equal(&scratch, &castedPred, &ybatch);
    Tensor *compMask = &compMaskVal;
    shapes_MoveTensorToHost(hostCtx, compMask);
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
    olib_ResetArena(scratch.memory);
    shapescuda_RewindMemory(&scratch.cudaMemory);
  }
}

void vgg10() {
  shapes_Context hostCtx = shapes_InitializeHostContext(25 * GB, 1);
  shapes_Context cudaCtx = shapes_InitializeCudaContext(10 * GB);

  dataset ds = getDataset(&hostCtx);

  shapesnn_FowardPassOp model = runTraining(&hostCtx, &cudaCtx, ds);
  // shapesnn_FowardPassOp model = Make_Model(&cudaCtx, 10);
  // shapesnn_LoadFromSafeTensors(&hostCtx, &model, "vgg10.safetensors");
  runInference(&hostCtx, &model, ds);
}

int main() {
  vgg10();

  return 0;
}
