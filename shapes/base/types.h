#ifndef shapes_types_h
#define shapes_types_h

#include <stdbool.h>
#include <stddef.h>
#include "array.h"
#include "memory.h"
#include "olib.h"
#include "shapes_common_types.h"
#include "shapescuda.h"
#ifdef SHAPES_HAS_CUDA 
  #include <cublas_v2.h>
#endif

typedef size_t shapes_dim_t;
typedef size_t shapes_multiplier_t;
typedef size_t shapes_tensor_size_t;

typedef struct {
  shapes_dim_t *dims;
  shapes_multiplier_t *multipliers;
  u8 numOfDims;
} shapes_Dim;

typedef struct ValuePair {
  shapes_Value a;
  shapes_Value b;
} valuePair;

typedef struct sizeAndMultipliers {
  size_t size;
  shapes_multiplier_t *multipliers;
} sizeAndMultipliers;

typedef enum { CPU, CUDA } DeviceType;

typedef struct {
  DeviceType type;
  char *id;
} shapes_Device;

typedef struct shapes_Context {
  olib_Memory *memory;
  olib_Memory *cudaMetadataMemory;
  shapescuda_Memory cudaMemory;
  shapes_Device *device;
  bool isTraining;

  #ifdef SHAPES_HAS_CUDA
  cublasHandle_t handle;
  #endif
  struct shapes_Context *parent;
} shapes_Context;

typedef struct shapes_Tensor {
  shapes_Context *context;
  string label;
  olib_Memory *metadataMemory;
  void *values;
  shapes_Range *boundary;
  size_t size;
  shapes_Dim shape;
  shapes_Dtype dtype;
  bool isView;
  bool isContigous;
  bool isContigousCopy;
  struct shapes_Tensor *grad;
  olib_Array *inputs;
  shapes_OpType opType;
  void *opMetadata;
  u64 nodeId;
} shapes_Tensor;

typedef struct {
  void *param;
  void *grad;
  void *m;
  void *v;
  size_t size;
  shapes_Dtype dtype;
} shapes_AdamData;

typedef struct shapes_BatchNormFowardResult {
  shapes_Tensor out;
  shapes_Tensor mean;
  shapes_Tensor variance;
} shapes_BatchNormFowardResult;

typedef struct shapes_BatchNormBackwardResult {
  shapes_Tensor dx2d;
  shapes_Tensor dGamma;
  shapes_Tensor dBeta;
} shapes_BatchNormBackwardResult;

typedef struct {
  shapes_Tensor a;
  shapes_Tensor b;
} shapes_TensorPair;

size_t getBytesForDtype(shapes_Dtype type);

#endif
