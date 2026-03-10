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

func DenseBackward(ctx Context, x Tensor, w Tensor, gradOut Tensor) (Tensor, Tensor, Tensor) {
	var dX *C.Tensor
	var dW *C.Tensor
	var dB *C.Tensor

	result := C.wrap_DenseBackward(
		(*C.Context)(ctx.UnsafePtr()),
		x.(*tensor).cTensor,
		w.(*tensor).cTensor,
		gradOut.(*tensor).cTensor,
		&dX,
		&dW,
		&dB,
	)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}

	return track(ctx, &tensor{cTensor: dX}),
		track(ctx, &tensor{cTensor: dW}),
		track(ctx, &tensor{cTensor: dB})
}

func BatchNormForwardTraining(ctx Context, x2d Tensor, gamma Tensor, beta Tensor, epsilon float32) (Tensor, Tensor, Tensor) {
	var out *C.Tensor
	var mean *C.Tensor
	var variance *C.Tensor

	result := C.wrap_BatchNormForwardTraining(
		(*C.Context)(ctx.UnsafePtr()),
		x2d.(*tensor).cTensor,
		gamma.(*tensor).cTensor,
		beta.(*tensor).cTensor,
		C.f32(epsilon),
		&out,
		&mean,
		&variance,
	)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}

	return track(ctx, &tensor{cTensor: out}),
		track(ctx, &tensor{cTensor: mean}),
		track(ctx, &tensor{cTensor: variance})
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

func Conv2d(ctx Context, x Tensor, kernels Tensor, stride uint8) Tensor {
	kernelShape := kernels.Shape()
	if len(kernelShape) < 4 {
		panic("shapes: Conv2d kernels must be 4D [outChannels,inChannels,kH,kW]")
	}

	outChannels := kernelShape[0]
	inChannels := kernelShape[1]
	kernelH := kernelShape[2]
	kernelW := kernelShape[3]

	var out *C.Tensor
	result := C.wrap_Conv2d(
		(*C.Context)(ctx.UnsafePtr()),
		C.size_t(inChannels),
		C.size_t(outChannels),
		C.u8(stride),
		kernels.(*tensor).cTensor,
		C.dim_t(kernelH),
		C.dim_t(kernelW),
		x.(*tensor).cTensor,
		&out,
	)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}

	return track(ctx, &tensor{cTensor: out})
}
