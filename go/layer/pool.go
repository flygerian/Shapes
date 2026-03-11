package layer

import "github.com/flygerian/shapes"

type maxPool struct {
	kernelShape shapes.Shape
	stride      uint8
}

type maxPoolMetadata struct {
	kernelShape shapes.Shape
	stride      uint8
}

type adaptiveAvgPool struct {
	outputSize shapes.Shape
}

type adaptiveAvgPoolMetadata struct {
	outputSize shapes.Shape
}

func MaxPooling(kernel shapes.Shape, stride uint8) Layer {
	if len(kernel) != 2 {
		panic("MaxPooling kernel shape should be 2D")
	}
	for _, s := range kernel {
		if s == 0 {
			panic("No dim of the kernel should be zero")
		}
	}

	return &maxPool{kernelShape: append(shapes.Shape(nil), kernel...), stride: stride}
}

func MaxPool2d(kernel shapes.Shape, stride uint8) Layer {
	return MaxPooling(kernel, stride)
}

func (m *maxPool) Forward(ctx shapes.Context, x shapes.Tensor) shapes.Tensor {
	kernelShape := append(shapes.Shape(nil), m.kernelShape...)
	forwardCtx := ctx.Forward(
		shapes.WithInputs(x),
		shapes.WithOpType(shapes.OpMaxPool),
		shapes.WithMetadata(maxPoolMetadata{kernelShape: kernelShape, stride: m.stride}),
	)

	out := shapes.MaxPool2d(forwardCtx, x, kernelShape, m.stride)
	forwardCtx.Finish(
		shapes.WithResult(out),
		shapes.WithBackward(maxPoolBackward),
	)

	return out
}

func maxPoolBackward(ctx shapes.Context, out shapes.ComputationGraphNode) {
	backwardCtx := ctx.Backward()
	defer backwardCtx.Finish()

	x := out.Inputs()[0]
	meta := out.Metadata().(maxPoolMetadata)
	dX := shapes.MaxPool2dBackward(backwardCtx, x, out.Grad().(shapes.Tensor), meta.kernelShape, meta.stride)
	x.Grad().Accumulate(backwardCtx, dX)
}

func AdaptiveAvgPool(outputSize shapes.Shape) Layer {
	if len(outputSize) != 2 {
		panic("AdaptiveAvgPool output size should be 2D")
	}
	for _, s := range outputSize {
		if s == 0 {
			panic("No dim of the output size should be zero")
		}
	}

	return &adaptiveAvgPool{outputSize: append(shapes.Shape(nil), outputSize...)}
}

func AdaptiveAvgPool2d(outputSize shapes.Shape) Layer {
	return AdaptiveAvgPool(outputSize)
}

func (a *adaptiveAvgPool) Forward(ctx shapes.Context, x shapes.Tensor) shapes.Tensor {
	outputSize := append(shapes.Shape(nil), a.outputSize...)
	forwardCtx := ctx.Forward(
		shapes.WithInputs(x),
		shapes.WithOpType(shapes.OpAdaptiveAvgPool),
		shapes.WithMetadata(adaptiveAvgPoolMetadata{outputSize: outputSize}),
	)

	out := shapes.AdaptiveAvgPool2d(forwardCtx, x, outputSize)
	forwardCtx.Finish(
		shapes.WithResult(out),
		shapes.WithBackward(adaptiveAvgPoolBackward),
	)

	return out
}

func adaptiveAvgPoolBackward(ctx shapes.Context, out shapes.ComputationGraphNode) {
	backwardCtx := ctx.Backward()
	defer backwardCtx.Finish()

	x := out.Inputs()[0]
	meta := out.Metadata().(adaptiveAvgPoolMetadata)
	dX := shapes.AdaptiveAvgPool2dBackward(backwardCtx, x, out.Grad().(shapes.Tensor), meta.outputSize)
	x.Grad().Accumulate(backwardCtx, dX)
}
