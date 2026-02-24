package main

import (
	"context"

	"github.com/flygerian/shapes"
	"github.com/flygerian/shapes/examples"
)

const Mb = 1024 * 1024

func main() {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true), shapes.WithArenaSize(100*Mb))
	defer ctx.Close()

	examples.MakeMore_2(ctx)
	// examples.MakeMore_1(ctx)
}
