package optimizer

/*
#cgo CFLAGS: -I../../base
#cgo LDFLAGS: -L../../base/build -L../../base/build/openblas/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "shapes.h"
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

		// Zero hidden state (parameter) gradients — these are not graph nodes themselves
		// so they are not visited by the loop above, but they accumulate gradients via backward.
		for _, p := range node.HiddenState() {
			if p.Grad() != nil {
				p.SetValuesToZero()
			}
		}
	}
}
