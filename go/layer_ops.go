package shapes

/*
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

func Conv2d(ctx Context, x Tensor, kernels Tensor, bias Tensor, withBias bool, stride uint8) (Tensor, Tensor) {
	kernelShape := kernels.Shape()
	if len(kernelShape) < 4 {
		panic("shapes: Conv2d kernels must be 4D [outChannels,inChannels,kH,kW]")
	}
	if withBias && bias == nil {
		panic("shapes: Conv2d requires bias tensor when withBias is true")
	}

	outChannels := kernelShape[0]
	inChannels := kernelShape[1]
	var biasTensor *C.Tensor
	if bias != nil {
		biasTensor = bias.(*tensor).cTensor
	}

	var out *C.Tensor
	var colBuffer *C.Tensor

	result := C.wrap_Conv2d(
		(*C.Context)(ctx.UnsafePtr()),
		C.size_t(inChannels),
		C.size_t(outChannels),
		C.u8(stride),
		kernels.(*tensor).cTensor,
		biasTensor,
		C.bool(withBias),
		x.(*tensor).cTensor,
		&out,
		&colBuffer,
	)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}

	return track(ctx, &tensor{cTensor: out}), track(ctx, &tensor{cTensor: colBuffer})
}

func Conv2dBackward(ctx Context, x Tensor, kernels Tensor, outputGrad Tensor, colBuffer Tensor, dBias Tensor, withBias bool, stride uint8) {
	var dBiasTensor *C.Tensor
	if dBias != nil {
		dBiasTensor = dBias.(*tensor).cTensor
	}

	result := C.wrap_Conv2dBackward(
		(*C.Context)(ctx.UnsafePtr()),
		x.(*tensor).cTensor,
		x.Grad().(*tensor).cTensor,
		kernels.(*tensor).cTensor,
		kernels.Grad().(*tensor).cTensor,
		outputGrad.(*tensor).cTensor,
		colBuffer.(*tensor).cTensor,
		dBiasTensor,
		C.bool(withBias),
		C.u8(stride),
	)

	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
}

func ConvTranspose2d(ctx Context, x Tensor, kernels Tensor, stride uint8) Tensor {
	kernelShape := kernels.Shape()
	if len(kernelShape) < 4 {
		panic("shapes: ConvTranspose2d kernels must be 4D [inChannels,outChannels,kH,kW]")
	}

	inChannels := kernelShape[0]
	outChannels := kernelShape[1]
	kernelH := kernelShape[2]
	kernelW := kernelShape[3]

	var out *C.Tensor
	result := C.wrap_ConvTranspose2d(
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

func ConvTranspose2dBackward(ctx Context, x Tensor, kernels Tensor, gradOut Tensor, stride uint8) (Tensor, Tensor) {
	var dX *C.Tensor
	var dKernels *C.Tensor

	result := C.wrap_ConvTranspose2dBackward(
		(*C.Context)(ctx.UnsafePtr()),
		x.(*tensor).cTensor,
		kernels.(*tensor).cTensor,
		gradOut.(*tensor).cTensor,
		C.u8(stride),
		&dX,
		&dKernels,
	)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}

	return track(ctx, &tensor{cTensor: dX}),
		track(ctx, &tensor{cTensor: dKernels})
}

func MaxPool2d(ctx Context, x Tensor, kernel Shape, stride uint8) Tensor {
	if len(kernel) != 2 {
		panic("shapes: MaxPool2d kernel must be 2D [kH,kW]")
	}

	var out *C.Tensor
	result := C.wrap_MaxPool2d(
		(*C.Context)(ctx.UnsafePtr()),
		x.(*tensor).cTensor,
		C.dim_t(kernel[0]),
		C.dim_t(kernel[1]),
		C.u8(stride),
		&out,
	)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}

	return track(ctx, &tensor{cTensor: out})
}

func MaxPool2dWithIndices(ctx Context, x Tensor, kernel Shape, stride uint8) (Tensor, Tensor) {
	if len(kernel) != 2 {
		panic("shapes: MaxPool2dWithIndices kernel must be 2D [kH,kW]")
	}

	var out *C.Tensor
	var indices *C.Tensor
	result := C.wrap_MaxPool2dWithIndices(
		(*C.Context)(ctx.UnsafePtr()),
		x.(*tensor).cTensor,
		C.dim_t(kernel[0]),
		C.dim_t(kernel[1]),
		C.u8(stride),
		&out,
		&indices,
	)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}

	return track(ctx, &tensor{cTensor: out}), track(ctx, &tensor{cTensor: indices})
}

func MaxPool2dBackward(ctx Context, x Tensor, gradOut Tensor, kernel Shape, stride uint8) Tensor {
	if len(kernel) != 2 {
		panic("shapes: MaxPool2dBackward kernel must be 2D [kH,kW]")
	}

	var dX *C.Tensor
	result := C.wrap_MaxPool2dBackward(
		(*C.Context)(ctx.UnsafePtr()),
		x.(*tensor).cTensor,
		gradOut.(*tensor).cTensor,
		C.dim_t(kernel[0]),
		C.dim_t(kernel[1]),
		C.u8(stride),
		&dX,
	)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}

	return track(ctx, &tensor{cTensor: dX})
}

func MaxPool2dBackwardWithIndices(ctx Context, x Tensor, gradOut Tensor, indices Tensor) Tensor {
	var dX *C.Tensor
	result := C.wrap_MaxPool2dBackwardWithIndices(
		(*C.Context)(ctx.UnsafePtr()),
		x.(*tensor).cTensor,
		gradOut.(*tensor).cTensor,
		indices.(*tensor).cTensor,
		&dX,
	)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}

	return track(ctx, &tensor{cTensor: dX})
}

func AdaptiveAvgPool2d(ctx Context, x Tensor, outputSize Shape) Tensor {
	if len(outputSize) != 2 {
		panic("shapes: AdaptiveAvgPool2d output size must be 2D [outH,outW]")
	}

	var out *C.Tensor
	result := C.wrap_AdaptiveAvgPool2d(
		(*C.Context)(ctx.UnsafePtr()),
		x.(*tensor).cTensor,
		C.dim_t(outputSize[0]),
		C.dim_t(outputSize[1]),
		&out,
	)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}

	return track(ctx, &tensor{cTensor: out})
}

func AdaptiveAvgPool2dBackward(ctx Context, x Tensor, gradOut Tensor, outputSize Shape) Tensor {
	if len(outputSize) != 2 {
		panic("shapes: AdaptiveAvgPool2dBackward output size must be 2D [outH,outW]")
	}

	var dX *C.Tensor
	result := C.wrap_AdaptiveAvgPool2dBackward(
		(*C.Context)(ctx.UnsafePtr()),
		x.(*tensor).cTensor,
		gradOut.(*tensor).cTensor,
		C.dim_t(outputSize[0]),
		C.dim_t(outputSize[1]),
		&dX,
	)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}

	return track(ctx, &tensor{cTensor: dX})
}
