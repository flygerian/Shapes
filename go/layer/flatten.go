package layer

import shapes "github.com/flygerian/shapes"

type flatten struct{}

func Flatten() *flatten {
	return &flatten{}
}

func (f *flatten) Forward(ctx shapes.Context, x shapes.Tensor) shapes.Tensor {
	shape := x.Shape()
	if len(shape) < 2 {
		return x
	}

	flattenedSize := 1
	for _, dim := range shape[1:] {
		flattenedSize *= int(dim)
	}

	return x.Reshape(ctx, int(shape[0]), flattenedSize)
}
