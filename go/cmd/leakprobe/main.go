package main

import (
	"context"
	"flag"
	"fmt"

	"github.com/flygerian/shapes"
)

const Mb = 1024 * 1024

func main() {
	mode := flag.String("mode", "reshape", "probe mode: reshape|reshape2|slice|transpose|squeeze|squeezedim|squeezedim1d|unsqueeze|noshape")
	epochs := flag.Int("epochs", 60, "number of epochs")
	flag.Parse()

	ctx := shapes.New(context.Background(), shapes.WithGrad(true), shapes.WithArenaSize(128*Mb))
	defer ctx.Finish()

	for i := 0; i < *epochs; i++ {
		e := ctx.Epoch(i)
		before := ctx.NumAllocatedBlocks()

		switch *mode {
		case "reshape":
			x := shapes.Float(e, shapes.Shape{32, 3, 2}, 1)
			_ = x.Reshape(e, -1, 6)
		case "reshape2":
			x := shapes.Float(e, shapes.Shape{32, 3, 2}, 1)
			y := x.Reshape(e, -1, 6)
			_ = y.Reshape(e, 32, 3, 2)
		case "slice":
			x := shapes.Float(e, shapes.Shape{4, 4}, 1)
			_ = x.Slice(e, shapes.Range{1, 3}, shapes.Range{0, 2})
		case "transpose":
			x := shapes.Float(e, shapes.Shape{3, 2}, 1)
			_ = x.Transpose(e, 0, 1)
		case "squeeze":
			x := shapes.Float(e, shapes.Shape{1, 3, 1, 2}, 1)
			_ = x.Squeeze(e)
		case "squeezedim":
			x := shapes.Float(e, shapes.Shape{1, 3, 2}, 1)
			_ = x.SqueezeDim(e, 0)
		case "squeezedim1d":
			x := shapes.Float(e, shapes.Shape{1}, 1)
			_ = x.SqueezeDim(e, 0)
		case "unsqueeze":
			x := shapes.Float(e, shapes.Shape{3, 2}, 1)
			_ = x.UnSqueeze(e, 1)
		case "noshape":
			_ = shapes.Float(e, shapes.Shape{32, 3, 2}, 1)
		default:
			panic("unknown mode")
		}

		e.Finish()
		after := ctx.NumAllocatedBlocks()
		fmt.Printf("[%s %03d] before=%d after=%d delta=%+d\n", *mode, i, before, after, after-before)
	}
}
