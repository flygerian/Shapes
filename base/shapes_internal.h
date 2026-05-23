
#ifndef shapes_internal_h
#define shapes_internal_h

#include "utils_lib/cuda_memory.h"
#include "cuda_compat.h"

typedef enum { CPU, CUDA } DeviceType;

typedef struct {
  DeviceType type;
  char *id;
} Device;

typedef struct Context {
  Memory *memory;
  Memory *cudaMetadataMemory;
  CudaMemory cudaMemory;
  Device *device;
  bool isTraining;
  cublasHandle_t handle;
  struct Context *parent;
} Context;

void attachHostDevice(Context *ctx);
void attachCudaDevice(Context *ctx);

#endif
