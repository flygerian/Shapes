package layer

import "github.com/flygerian/shapes"

type Layer interface {
	Forward(ctx shapes.Context, x shapes.Tensor) shapes.Tensor
}
