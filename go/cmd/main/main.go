package main

import (
	"context"
	"fmt"

	"github.com/flygerian/shapes"
	"github.com/flygerian/shapes/tensor"
)

func main() {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	// Create a 2x3 tensor filled with 3.14
	t := tensor.Float(ctx, []uint32{2, 3}, 3.14)
	fmt.Printf("Created tensor: %v\n", t)

	zeros := tensor.Zeros(ctx, tensor.Shape{2, 3})
	fmt.Printf("Created zeros: %v\n", zeros)
}
