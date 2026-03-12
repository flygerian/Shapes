package shapes

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/build/openblas/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "cwrappers.h"
*/
import "C"

import "unsafe"

// Sgd performs stochastic gradient descent optimization on parameters using their gradients.
// It calls the optimized C implementation directly.
func Sgd(ctx Context, parameters []Tensor, grads []Tensor, learningRate float32) {
	if len(parameters) == 0 {
		return
	}
	if len(parameters) != len(grads) {
		panic("shapes: Sgd requires equal number of parameters and gradients")
	}

	cParams := make([]*C.Tensor, len(parameters))
	cGrads := make([]*C.Tensor, len(grads))

	for i := range parameters {
		cParams[i] = parameters[i].(*tensor).cTensor
		cGrads[i] = grads[i].(*tensor).cTensor
	}

	result := C.Sgd(
		(*C.Context)(ctx.UnsafePtr()),
		(**C.Tensor)(unsafe.Pointer(&cParams[0])),
		(**C.Tensor)(unsafe.Pointer(&cGrads[0])),
		C.size_t(len(cParams)),
		C.f32(learningRate),
	)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
}

// AdamData represents a single parameter's data for the Adam optimizer.
type AdamData struct {
	Param     Tensor
	ParamGrad Tensor
	M         Tensor
	V         Tensor
}

// Adam performs Adam optimization on parameters using their gradients.
// It calls the optimized C implementation directly.
func Adam(ctx Context, data []AdamData, beta1, beta2 float32, step int, learningRate, epsilon float32) {
	if len(data) == 0 {
		return
	}

	cData := make([]C.AdamData, len(data))
	for i, d := range data {
		cData[i] = C.AdamData{
			param:     d.Param.(*tensor).cTensor,
			paramGrad: d.ParamGrad.(*tensor).cTensor,
			m:         d.M.(*tensor).cTensor,
			v:         d.V.(*tensor).cTensor,
		}
	}

	result := C.Adam(
		(*C.Context)(ctx.UnsafePtr()),
		(*C.AdamData)(unsafe.Pointer(&cData[0])),
		C.size_t(len(cData)),
		C.f32(beta1),
		C.f32(beta2),
		C.size_t(step),
		C.f32(learningRate),
		C.f32(epsilon),
	)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
}
