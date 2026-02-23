package optimizer

/*
#cgo CFLAGS: -I../../base
#cgo LDFLAGS: -L../../base/build -L../../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "tensor/tensor.h"
#include "common.h"
#include <string.h>

static inline void zeroTensorValues(Tensor *t) {
	if (t != NULL && t->values != NULL) {
		size_t bytes = t->size * getBytesForDtype(t->dtype);
		memset(t->values, 0, bytes);
	}
}
*/
import "C"

import (
	shapes "github.com/flygerian/shapes"
)

func ZeroGrad(ctx *shapes.Context, cg *shapes.ComputationGraph) {
	for _, node := range cg.Nodes {
		// Instead of allocating a new zero tensor, just zero out the existing gradient's values
		if node.Grad != nil {
			C.zeroTensorValues((*C.Tensor)(node.Grad.UnsafeCTensor()))
		}
	}
}
