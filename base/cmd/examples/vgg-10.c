#include "result/result.h"
#include "shapes.h"
#include "tensor/types.h"
#include "utils_lib/array.h"
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

typedef struct dataset {
  Array_Tensor Xs;
  Array_Tensor Ys;
} dataset;

int globErrFn(const char* epath, int eerrno) {
  printf("Err in path %s errno %d", epath, eerrno);
}

dataset getDataset(Context *ctx) { 
  glob_t batchFiles; 
  i32 err = glob(DATASET_PATH, GLOB_ERR, globErrFn, &batchFiles); 

  PANIC_WITH_MSG_IF(err != 0, "could error while globbing dataset path");

  File labelsDataFile  = File_OpenPathInReadMode(LABELS_FILE_PATH);
  File testBatchFile = File_OpenPathInReadMode( TEST_BATCH_FILE);

  Array_Tensor Xs = Make_DynamicTensorArray(ctx->memory);
  Array_Tensor Ys = Make_DynamicTensorArray(ctx->memory);

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
      Error readErr = File_ReadBytesToBuffer(&bf, &label, 1);
      if(readErr.code != 0) {
        break;
      }

      // Read the R channel
      readErr = File_ReadBytesToBuffer(&bf, r, CHANNEL_PLANE);
      if(readErr.code != 0) {
        break;
      }

      // Read the B channel
      readErr = File_ReadBytesToBuffer(&bf, g, CHANNEL_PLANE);
      if(readErr.code != 0) {
        break;
      }

      // Read the C channel
      readErr = File_ReadBytesToBuffer(&bf, b,CHANNEL_PLANE);
      if(readErr.code != 0) {
        break;
      }

      for (RANGE(pixelIdx, CHANNEL_PLANE)) {
        f32 pixel[3] = {F32_(r[pixelIdx]) / 255,  F32_(g[pixelIdx]) / 255, F32_(b[pixelIdx]) / 255};
        Array_AppendF32Buffer(image, pixel, 3);
      }

      Tensor* imageTensor = MakeFromContigousArray(ctx, SHAPE3D(IMAGES_HEIGHT, IMAGES_WIDTH, NUM_CHANNELS), image->items, F32);
      Array_AppendTensor(Xs, imageTensor);

      f32 lbl[1] = {F32_(label[0])}; 
      Tensor* labelTensor =  MakeFromContigousArray(ctx, SCALAR, lbl, F32);
      Array_AppendTensor(Ys, labelTensor);

      Array_Reset(image);
      numImagesProcessed += 1;

      printf("\nProcessed %zu images", numImagesProcessed);
    }

    CloseFile(&bf);
  }

  return (dataset) {.Xs = Xs, .Ys = Ys};
}

void vgg10() {

  Memory *mem = initializeArena((size_t)1024 * 1024 * 1024 * 5, 1); // 5GB
  Context ctx = {.memory = mem};

  dataset ds = getDataset(&ctx);
  Tensor *first = Array_TensorIdx(ds.Xs, 234);
  basicRaylibWindow(first);
}
