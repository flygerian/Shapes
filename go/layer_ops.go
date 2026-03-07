package shapes

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/build/openblas/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "cwrappers.h"
*/
import "C"

func DenseLinear(ctx Context, x Tensor, w Tensor, b Tensor, withBias bool) Tensor {
	if withBias && b == nil {
		panic("shapes: DenseLinear requires bias tensor when withBias is true")
	}

	var bTensor *C.Tensor
	if b != nil {
		bTensor = b.(*tensor).cTensor
	}

	var dest *C.Tensor
	result := C.wrap_DenseLinear(
		(*C.Context)(ctx.UnsafePtr()),
		x.(*tensor).cTensor,
		w.(*tensor).cTensor,
		bTensor,
		C.bool(withBias),
		&dest,
	)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}

	return track(ctx, &tensor{cTensor: dest})
}

func BatchNormBackward(ctx Context, x2d Tensor, grad2d Tensor, gamma Tensor, epsilon float32) (Tensor, Tensor, Tensor) {
	var dX *C.Tensor
	var dGamma *C.Tensor
	var dBeta *C.Tensor

	result := C.wrap_BatchNormBackward(
		(*C.Context)(ctx.UnsafePtr()),
		x2d.(*tensor).cTensor,
		grad2d.(*tensor).cTensor,
		gamma.(*tensor).cTensor,
		C.f32(epsilon),
		&dX,
		&dGamma,
		&dBeta,
	)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}

	return track(ctx, &tensor{cTensor: dX}),
		track(ctx, &tensor{cTensor: dGamma}),
		track(ctx, &tensor{cTensor: dBeta})
}
