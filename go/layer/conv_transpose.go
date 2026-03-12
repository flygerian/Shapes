package layer

import (
	"math"

	"github.com/flygerian/shapes"
)

type convTranspose struct {
	kernelShape shapes.Shape
	stride      uint8
	inChannels  uint
	outChannels uint

	kernels       shapes.Tensor
	bias          shapes.Tensor
	isBiasApplied bool
}

func ConvTranspose2d(
	ctx shapes.Context,
	inChannels uint,
	outChannels uint,
	kernel shapes.Shape,
	stride uint8,
	options ...layerOption,
) Layer {
	if inChannels == 0 || outChannels == 0 {
		panic("Cannot have in/out channnels as 0")
	}
	if len(kernel) != 2 {
		panic("ConvTranspose2d kernel shape should be 2D")
	}

	for _, s := range kernel {
		if s == 0 {
			panic("No dim of the kernel should be zero")
		}
	}

	initialization := (5 / 3) / (math.Pow(float64(outChannels), 0.5))
	kernels := shapes.FloatRandom(
		ctx,
		shapes.Shape{uint(inChannels), uint(outChannels), kernel[0], kernel[1]},
		-float32(initialization), float32(initialization),
	)
	bias := shapes.Float(ctx, shapes.Shape{uint(outChannels)}, 0)

	state := convTranspose{
		kernelShape: kernel,
		inChannels:  inChannels,
		outChannels: outChannels,
		kernels:     kernels,
		stride:      stride,
		bias:        bias,
	}

	for _, opt := range options {
		opt(&state)
	}

	return &state
}

func (c *convTranspose) SetBiasEnabled(isBiasEnabled bool) {
	c.isBiasApplied = isBiasEnabled
}

func (c *convTranspose) Forward(ctx shapes.Context, x shapes.Tensor) shapes.Tensor {
	forwardCtx := ctx.Forward(
		shapes.WithInputs(x),
		shapes.WithHiddenState(c.kernels, c.bias),
		shapes.WithOpType(shapes.OpConvTranspose),
		shapes.WithMetadata(convMetadata{stride: c.stride}),
	)

	out := shapes.ConvTranspose2d(forwardCtx, x, c.kernels, c.stride).Plus(forwardCtx, c.bias)

	forwardCtx.Finish(
		shapes.WithResult(out),
		shapes.WithBackward(convTransposeBackward),
	)

	return out
}

func convTransposeBackward(ctx shapes.Context, out shapes.ComputationGraphNode) {
	backwardCtx := ctx.Backward()
	defer backwardCtx.Finish()

	x := out.Inputs()[0]
	hidden := out.HiddenState()
	kernels := hidden[0]
	bias := hidden[1]
	meta := out.Metadata().(convMetadata)

	dOutput := out.Grad()
	outputShape := dOutput.Shape()

	B := outputShape[0]
	C := outputShape[1]
	oh := outputShape[2]
	ow := outputShape[3]

	dBias := dOutput.Reshape(backwardCtx, int(B), int(C), int(oh*ow)).Sum(backwardCtx, 2).Sum(backwardCtx, 0).Squeeze(backwardCtx)
	bias.Grad().Accumulate(backwardCtx, dBias)

	dX, dKernels := shapes.ConvTranspose2dBackward(backwardCtx, x, kernels, dOutput.(shapes.Tensor), meta.stride)
	x.Grad().Accumulate(backwardCtx, dX)
	kernels.Grad().Accumulate(backwardCtx, dKernels)
}
