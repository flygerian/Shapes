#ifndef shapes_common_h
#define shapes_common_h

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "cuda_compat.h"
#include <termios.h>

#define SHAPE(dimensions, numberOfDimensions)           ((Dim){.dims = (dimensions), .numOfDims = (numberOfDimensions)})
#define SCALAR                                          ((Dim){.dims = (dim_t[]){1}, .numOfDims = 1})
#define SHAPE1D(dimSize)                                ((Dim){.dims = (dim_t[]){dimSize}, .numOfDims = 1})
#define SHAPE2D(dim0Size, dim1Size)                     ((Dim){.dims = (dim_t[]){dim0Size, dim1Size}, .numOfDims = 2})
#define SHAPE3D(dim0Size, dim1Size, dim2Size)           ((Dim){.dims = (dim_t[]){dim0Size, dim1Size, dim2Size}, .numOfDims = 3})
#define SHAPE4D(dim0Size, dim1Size, dim2Size, dim3Size) ((Dim){.dims = (dim_t[]){dim0Size, dim1Size, dim2Size, dim3Size}, .numOfDims = 4})
#endif
