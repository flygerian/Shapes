package layer

import "github.com/flygerian/shapes"

type maxPool struct {
	ctx         shapes.Context
	kernelShape shapes.Shape
	stride      uint8
}

type maxPoolMetadata struct {
	kernelShape shapes.Shape
	stride      uint8
	indices     shapes.Tensor
}

type adaptiveAvgPool struct {
	ctx        shapes.Context
	outputSize shapes.Shape
}

type adaptiveAvgPoolMetadata struct {
	outputSize shapes.Shape
}

func MaxPooling(ctx shapes.Context, kernel shapes.Shape, stride uint8) *maxPool {
	if len(kernel) != 2 {
		panic("MaxPooling kernel shape should be 2D")
	}
	for _, s := range kernel {
		if s == 0 {
			panic("No dim of the kernel should be zero")
		}
	}

	return &maxPool{ctx: ctx, kernelShape: append(shapes.Shape(nil), kernel...), stride: stride}
}

func MaxPool2d(ctx shapes.Context, kernel shapes.Shape, stride uint8) *maxPool {
	return MaxPooling(ctx, kernel, stride)
}

func (m *maxPool) Forward(ctx shapes.Context, x shapes.Tensor) shapes.Tensor {
	kernelShape := append(shapes.Shape(nil), m.kernelShape...)
	forwardCtx := ctx.Forward(
		shapes.WithInputs(x),
		shapes.WithOpType(shapes.OpMaxPool),
	)

	out, indices := shapes.MaxPool2dWithIndices(forwardCtx, x, kernelShape, m.stride)
	forwardCtx.Finish(
		shapes.WithResult(out),
		shapes.WithBackward(maxPoolBackward),
		shapes.WithMetadata(maxPoolMetadata{kernelShape: kernelShape, stride: m.stride, indices: indices}),
	)

	return out
}

func maxPoolBackward(ctx shapes.Context, out shapes.ComputationGraphNode) {
	backwardCtx := ctx.Backward()
	defer backwardCtx.Finish()

	x := out.Inputs()[0]
	meta := out.Metadata().(maxPoolMetadata)
	dX := shapes.MaxPool2dBackwardWithIndices(backwardCtx, x, out.Grad().(shapes.Tensor), meta.indices)
	x.Grad().Accumulate(backwardCtx, dX)

	backwardCtx.Mark(meta.indices)
}

func AdaptiveAvgPool(ctx shapes.Context, outputSize shapes.Shape) *adaptiveAvgPool {
	if len(outputSize) != 2 {
		panic("AdaptiveAvgPool output size should be 2D")
	}
	for _, s := range outputSize {
		if s == 0 {
			panic("No dim of the output size should be zero")
		}
	}

	return &adaptiveAvgPool{ctx: ctx, outputSize: append(shapes.Shape(nil), outputSize...)}
}

func AdaptiveAvgPool2d(ctx shapes.Context, outputSize shapes.Shape) *adaptiveAvgPool {
	return AdaptiveAvgPool(ctx, outputSize)
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
