package optimizer

/*
#cgo CFLAGS: -I../../base
#cgo LDFLAGS: -L../../base/build -L../../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "tensor/tensor.h"
#include "common.h"
#include <string.h>

*/
import "C"

import (
	shapes "github.com/flygerian/shapes"
)

func ZeroGrad(ctx shapes.Context, cg shapes.ComputationGraph) {
	for _, node := range cg {
		// Instead of allocating a new zero tensor, just zero out the existing gradient's values
		if node.Grad() != nil {
			node.SetValuesToZero()
		}
	}
}
