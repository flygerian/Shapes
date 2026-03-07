package optimizer

/*
#cgo CFLAGS: -I../../base
#cgo LDFLAGS: -L../../base/build -L../../base/build/openblas/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "shapes.h"
*/
import "C"

import (
	shapes "github.com/flygerian/shapes"
	"github.com/flygerian/shapes/extract"
	"unsafe"
)

func SGD(ctx shapes.Context, lr float32) func(shapes.ComputationGraph) {
	return func(cg shapes.ComputationGraph) {
		fusedCtx := ctx.NoGraph()
		defer fusedCtx.Finish()

		parameters := extract.Parameters(cg)
		if len(parameters) == 0 {
			return
		}

		cParams := make([]*C.Tensor, len(parameters))
		cGrads := make([]*C.Tensor, len(parameters))

		for i, p := range parameters {
			paramPtr, ok := p.(interface{ UnsafeCTensor() unsafe.Pointer })
			if !ok {
				panic("shapes: parameter does not expose underlying C tensor")
			}
			grad := p.Grad()
			if grad == nil {
				panic("shapes: parameter gradient is nil")
			}
			gradPtr, ok := grad.(interface{ UnsafeCTensor() unsafe.Pointer })
			if !ok {
				panic("shapes: gradient does not expose underlying C tensor")
			}

			cParams[i] = (*C.Tensor)(paramPtr.UnsafeCTensor())
			cGrads[i] = (*C.Tensor)(gradPtr.UnsafeCTensor())
		}

		result := C.Sgd(
			(*C.Context)(fusedCtx.UnsafePtr()),
			(**C.Tensor)(unsafe.Pointer(&cParams[0])),
			(**C.Tensor)(unsafe.Pointer(&cGrads[0])),
			C.size_t(len(cParams)),
			C.f32(lr),
		)
		if result != C.OK {
			panic("shapes: " + shapes.ResultString(uint32(result)))
		}
	}
}
