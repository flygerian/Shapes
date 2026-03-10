package layer

import (
	"math"

	"github.com/flygerian/shapes"
)

type conv struct {
	kernelShape shapes.Shape
	stride      uint8
	inChannels  uint
	outChannels uint

	kernels shapes.Tensor
	bias    shapes.Tensor
}

func Conv2d(ctx shapes.MainContext, inChannels uint, outChannels uint, kernel shapes.Shape) func(shapes.Context, shapes.Tensor) {
	if inChannels == 0 || outChannels == 0 {
		panic("Cannot have in/out channnels as 0")
	}

	for _, s := range kernel {
		if s == 0 {
			panic("No dim of the kernel should be zero")
		}
	}

	initialization := (5 / 3) / (math.Pow(float64(inChannels), 0.5))
	kernels := shapes.FloatRandom(
		ctx,
		shapes.Shape{uint(outChannels), uint(inChannels), kernel[0], kernel[1]},
		-float32(initialization), float32(initialization),
	)
	bias := shapes.Float(ctx, shapes.Shape{uint(outChannels)}, 0)

	state := conv{
		kernelShape: kernel,
		inChannels:  inChannels,
		outChannels: outChannels,

		kernels: kernels,
		bias:    bias,
	}

	return func(ctx shapes.Context, x shapes.Tensor) {
		fowardCtx := ctx.Forward(
			shapes.WithInputs(x),
			shapes.WithHiddenState(state.kernels),
			shapes.WithOpType(shapes.OpConv),
		)

		shapes.Conv2d(ctx, x, state.kernels, state.stride)
	}
}
