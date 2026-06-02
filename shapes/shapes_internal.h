
#ifndef shapes_internal_h
#define shapes_internal_h

#ifdef SHAPES_ENABLE_CUDA 
#include "cuda_memory.h"
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "cuda_compat.h"
#include <termios.h>

typedef enum { CPU, CUDA } DeviceType;

typedef struct {
  DeviceType type;
  char *id;
} Device;

typedef struct Context {
  Memory *memory;
  Memory *cudaMetadataMemory;

  #ifdef SHAPES_ENABLE_CUDA 
  CudaMemory cudaMemory;
  #else
  void* cudaMemory;
  #endif

  Device *device;
  bool isTraining;
  cublasHandle_t handle;
  struct Context *parent;
} Context;

void attachHostDevice(Context *ctx);
void attachCudaDevice(Context *ctx);

#define SHAPE(dimensions, numberOfDimensions)           ((Dim){.dims = (dimensions), .numOfDims = (numberOfDimensions)})
#define SCALAR                                          ((Dim){.dims = (dim_t[]){1}, .numOfDims = 1})
#define SHAPE1D(dimSize)                                ((Dim){.dims = (dim_t[]){dimSize}, .numOfDims = 1})
#define SHAPE2D(dim0Size, dim1Size)                     ((Dim){.dims = (dim_t[]){dim0Size, dim1Size}, .numOfDims = 2})
#define SHAPE3D(dim0Size, dim1Size, dim2Size)           ((Dim){.dims = (dim_t[]){dim0Size, dim1Size, dim2Size}, .numOfDims = 3})
#define SHAPE4D(dim0Size, dim1Size, dim2Size, dim3Size) ((Dim){.dims = (dim_t[]){dim0Size, dim1Size, dim2Size, dim3Size}, .numOfDims = 4})

#endif
