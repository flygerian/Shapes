#include "common.h"
#include "result/result.h"
#include "shapes.h"
#include "tensor/types.h"
#include "utils_lib/array.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <glob.h>
#include "utils_lib/file.h"
#include "utils_lib/utils_lib.h"
#include "utils_lib/error.h"
#include "string.h"
#include "examples.h"

#define DATASET_PATH "base/cmd/examples/datasets/cifar-10-binary/cifar-10-batches-bin/data_batch_*.bin"
#define LABELS_FILE_PATH "base/cmd/examples/datasets/cifar-10-binary/cifar-10-batches-bin/batches.meta.txt"
#define TEST_BATCH_FILE "base/cmd/examples/datasets/cifar-10-binary/cifar-10-batches-bin/test_batch.bin"
#define IMAGES_WIDTH 32
#define IMAGES_HEIGHT 32
#define NUM_CHANNELS 3
#define CHANNEL_PLANE IMAGES_WIDTH * IMAGES_HEIGHT
#define BATCH_SIZE 32

typedef struct dataset {
  Array_Tensor Xtrain;
  Array_Tensor Ytrain;
  Array_Tensor Xtest;
  Array_Tensor Ytest;
} dataset;

int globErrFn(const char* epath, int eerrno) {
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
  readErr = File_ReadBytesToBuffer(imageFile, out_b,CHANNEL_PLANE);
  RETURN_ON_ERROR(readErr);

  return OK;
}

static inline void readAsNCHWIntoArray(Array *image, byte *restrict r, byte *restrict g, byte *restrict b) {
  for (RANGE(pixelIdx, CHANNEL_PLANE)) {
    f32 pixel[3] = {F32_(r[pixelIdx]) / 255,  F32_(g[pixelIdx]) / 255, F32_(b[pixelIdx]) / 255};
    Array_AppendF32Buffer(image, pixel, 3);
  }
}

dataset getDataset(Context *ctx) { 
  glob_t batchFiles; 
  i32 err = glob(DATASET_PATH, GLOB_ERR, globErrFn, &batchFiles); 

  PANIC_WITH_MSG_IF(err != 0, "could error while globbing dataset path");

  File labelsDataFile  = File_OpenPathInReadMode(LABELS_FILE_PATH);
  File testBatchFile = File_OpenPathInReadMode( TEST_BATCH_FILE);

  Array_Tensor Xtrain = Make_DynamicTensorArray(ctx->memory);
  Array_Tensor Ytrain = Make_DynamicTensorArray(ctx->memory);
  Array_Tensor Xtest = Make_DynamicTensorArray(ctx->memory);
  Array_Tensor Ytest = Make_DynamicTensorArray(ctx->memory);

  byte label[1]= {0};
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

    while(true) {
      Error err = readNextImageChannelsAndLabel(bf, label, r, g, b);
      if (err.code != 0) {
        break;
      } 

      readAsNCHWIntoArray(image, r, g, b);

      Tensor* imageTensor = MakeFromContigousArray(ctx, SHAPE3D(IMAGES_HEIGHT, IMAGES_WIDTH, NUM_CHANNELS), image->items, F32);
      Array_AppendTensor(Xtrain, imageTensor);

      f32 lbl[1] = {F32_(label[0])}; 
      Tensor* labelTensor =  MakeFromContigousArray(ctx, SCALAR, lbl, F32);
      Array_AppendTensor(Ytrain, labelTensor);

      Array_Reset(image);
      numImagesProcessed += 1;

      printf("\nProcessed %zu images", numImagesProcessed);
    }

    CloseFile(&bf);
  }

  while (true) { 
    Error err = readNextImageChannelsAndLabel(testBatchFile, label, r, g, b);
    if (err.code != 0) {
      break;
    } 

    readAsNCHWIntoArray(image, r, g, b);
    Tensor* imageTensor = MakeFromContigousArray(ctx, SHAPE3D(IMAGES_HEIGHT, IMAGES_WIDTH, NUM_CHANNELS), image->items, F32);
    Array_AppendTensor(Xtest, imageTensor);

    f32 lbl[1] = {F32_(label[0])}; 
    Tensor* labelTensor =  MakeFromContigousArray(ctx, SCALAR, lbl, F32);
    Array_AppendTensor(Ytest, labelTensor);
    Array_Reset(image);
  }

  CloseFile(&labelsDataFile);
  CloseFile(&testBatchFile);
  return (dataset) {.Xtrain = Xtrain, .Ytrain = Ytrain, .Xtest = Xtest, .Ytest = Ytest};
}

static inline dataset toBatches(Context *ctx, dataset ds) {
  size_t amountProcessed = 0;
  Array_Tensor currentXBatch = Make_TensorArray(ctx->memory, BATCH_SIZE);
  Array_Tensor currentYBatch = Make_TensorArray(ctx->memory, BATCH_SIZE);

  dataset batched = (dataset) { 
    .Xtrain = Make_DynamicTensorArray(ctx->memory), 
    .Xtest = Make_DynamicTensorArray(ctx->memory), 
    .Ytrain = ds.Ytrain,
    .Ytest = ds.Ytest
  };

  u8 counter = 0;
  while (true) {
    Tensor *tXTrain = Array_TensorIdx(ds.Xtrain, counter);
    Array_AppendTensor(currentXBatch, tXTrain); 

    Tensor *tYTrain = Array_TensorIdx(ds.Ytrain, counter);
    Array_AppendTensor(currentYBatch, tYTrain); 

    bool isAtEnd = ( amountProcessed + counter ) >= ds.Xtrain->size;
    if (counter == 31 || isAtEnd) {
      Tensor* batchedXTensor = Stack(ctx, currentXBatch);
      Tensor* batchedYTensor = Stack(ctx, currentYBatch);

      Array_AppendTensor(batched.Xtrain, batchedXTensor);
      Array_AppendTensor(batched.Ytrain, batchedYTensor);

      counter = 0;
      Array_Reset(currentXBatch);
      Array_Reset(currentYBatch);

      if (isAtEnd) {
        break;
      }
    }

    counter += 1;
    amountProcessed += 1;
  }

  return batched;
} 

void vgg10() {

  Memory *mem = initializeArena((size_t)1024 * 1024 * 1024 * 5, 1); // 5GB
  Context ctx = {.memory = mem};

  dataset ds = getDataset(&ctx);
  // Tensor *first = Array_TensorIdx(ds.Xtest, 246);
  // basicRaylibWindow(first);
  
  dataset batched = toBatches(&ctx, ds); 
  printf("\n%zu batches created from dataset %zu size", batched.Xtrain->size, ds.Xtrain->size);

}
