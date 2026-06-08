package main
import "core:fmt"
import "shapes"

main :: proc() {
	dims := []uint{2, 3}

	kb :: 1024
	mb :: 1024 * kb
	gb :: 1024 * mb

	ctx := shapes.InitializeHostContext(5 * mb, 1)
	tensor := shapes.MakeRandomTensor(&ctx, shapes.Shape2D(1, 3), -4, -3, .F32)
	context.user_ptr = &ctx


}

