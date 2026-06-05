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

typedef size_t dim_t;
typedef size_t multiplier_t;
typedef size_t tensor_size_t;

typedef struct {
  dim_t *dims;
  multiplier_t *multipliers;
  u8 numOfDims;
} Dim;

typedef struct ValuePair {
  shapes_Value a;
  shapes_Value b;
} ValuePair;

typedef struct sizeAndMultipliers {
  size_t size;
  multiplier_t *multipliers;
} sizeAndMultipliers;

typedef enum { CPU, CUDA } DeviceType;

typedef struct {
  DeviceType type;
  char *id;
} Device;

typedef struct shapes_Context {
  olib_Memory *memory;
  olib_Memory *cudaMetadataMemory;
  shapescuda_Memory cudaMemory;
  Device *device;
  bool isTraining;

  #ifdef SHAPES_HAS_CUDA
  cublasHandle_t handle;
  #endif
  struct shapes_Context *parent;
} shapes_Context;

typedef struct Tensor {
  shapes_Context *context;
  string label;
  olib_Memory *metadataMemory;
  void *values;
  shapes_Range *boundary;
  size_t size;
  Dim shape;
  shapes_Dtype dtype;
  bool isView;
  bool isContigous;
  bool isContigousCopy;
  struct Tensor *grad;
  olib_Array *inputs;
  shapes_OpType opType;
  void *opMetadata;
  u64 nodeId;
} Tensor;

typedef struct {
  void *param;
  void *grad;
  void *m;
  void *v;
  size_t size;
  shapes_Dtype dtype;
} AdamData;

typedef struct BatchNormFowardResult {
  Tensor out;
  Tensor mean;
  Tensor variance;
} BatchNormFowardResult;

typedef struct BatchNormBackwardResult {
  Tensor dx2d;
  Tensor dGamma;
  Tensor dBeta;
} BatchNormBackwardResult;

typedef struct {
  Tensor a;
  Tensor b;
} TensorPair;

size_t getBytesForDtype(shapes_Dtype type);

#endif
