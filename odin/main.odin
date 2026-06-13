package main
import "core:fmt"
import "shapes"
import nn "shapes/nn"

main :: proc() {
	dims := []uint{2, 3}

	kb :: 1024
	mb :: 1024 * kb
	gb :: 1024 * mb

	ctx := shapes.InitializeHostContext(5 * mb, 1)
	context.user_ptr = &ctx

	x := shapes.MakeRandomTensor(minValue = -4, maxValue = -3, shape = shapes.Shape2D(1, 3))

	denseLayer := nn.Dense(3, 100, false)
	out := nn.Forward(&denseLayer, &x)
	shapes.PrintTensor(out)
	nn.denseBackward(&out)

}

