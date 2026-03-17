package main

/*
#cgo CFLAGS: -I../../../base
#include "memory.h"
*/
import "C"

import (
	"context"
	"flag"
	"fmt"

	"github.com/flygerian/shapes"
	"github.com/flygerian/shapes/layer"
	"github.com/flygerian/shapes/loss_fns"
	"github.com/flygerian/shapes/optimizer"
)

const Mb = 1024 * 1024

func buildDataset(n int) ([][]int8, []int8) {
	x := make([][]int8, n)
	y := make([]int8, n)
	for i := range n {
		a := int8((i % 26) + 1)
		b := int8(((i / 3) % 26) + 1)
		c := int8(((i / 7) % 26) + 1)
		x[i] = []int8{a, b, c}
		y[i] = int8((i % 27))
	}
	return x, y
}

func memoryStats(ctx shapes.Context) (allocated uint64, capacity uint64, blocks int, freeBlocks int) {
	mem := (*C.Memory)(ctx.UnsafeMemory())
	if mem == nil {
		return 0, 0, 0, 0
	}

	return uint64(mem.allocated), uint64(mem.capacity), int(mem.numBlocks), int(mem.numFreeBlocks)
}

func main() {
	stage := flag.Int("stage", 5, "0:empty,1:gather,2:forward,3:scalar,4:backward,5:opt")
	forwardLevel := flag.Int("forward_level", 4, "0:reshape+l1,1:+l2,2:+onehot,3:+crossentropy,4:full")
	ceStep := flag.Int("ce_step", -1, "cross-entropy inline step (0..9), -1 disables inline CE")
	ceContext := flag.String("ce_context", "root", "context for fused cross-entropy: root|run")
	epochs := flag.Int("epochs", 80, "number of epochs")
	printEvery := flag.Int("print_every", 1, "print cadence in epochs")
	flag.Parse()

	ctx := shapes.New(context.Background(), shapes.WithGrad(true), shapes.WithArenaSize(512*Mb))
	defer ctx.Finish()

	xData, yData := buildDataset(4096)
	X := shapes.FromInt8(ctx, xData)
	Y := shapes.FromInt8(ctx, yData)

	embLayer := layer.Embedding(ctx, 27, 2)
	l1 := layer.Dense(ctx, 6, 100)
	l2 := layer.Dense(ctx, 100, 27)
	crossEntropy := loss_fns.CrossEntropy()
	sgd := optimizer.SGD(ctx, 0.01)

	forward := func(runCtx shapes.Context, xBatch shapes.Tensor, yBatch shapes.Tensor) shapes.Tensor {
		h := l1.Forward(runCtx, xBatch.Reshape(runCtx, -1, 6))
		if *forwardLevel == 0 {
			return h.Mean(runCtx)
		}
		logits := l2.Forward(runCtx, h)
		if *forwardLevel == 1 {
			return logits.Mean(runCtx)
		}
		yOneHot := shapes.OneHot(runCtx, yBatch, 27)
		if *forwardLevel == 2 {
			return yOneHot.Mean(runCtx)
		}
		if *ceStep >= 0 {
			classDim := uint(len(logits.Shape()) - 1)
			maxLogits := logits.Max(runCtx, classDim)
			if *ceStep == 0 {
				return maxLogits.Mean(runCtx)
			}

			shifted := logits.Minus(runCtx, maxLogits)
			if *ceStep == 1 {
				return shifted.Mean(runCtx)
			}

			exp := shifted.Exp(runCtx)
			if *ceStep == 2 {
				return exp.Mean(runCtx)
			}

			sumExp := exp.Sum(runCtx, classDim)
			if *ceStep == 3 {
				return sumExp.Mean(runCtx)
			}

			probs := exp.Divide(runCtx, sumExp)
			if *ceStep == 4 {
				return probs.Mean(runCtx)
			}

			logProbs := probs.Log(runCtx)
			if *ceStep == 5 {
				return logProbs.Mean(runCtx)
			}

			times := yOneHot.Times(runCtx, logProbs)
			if *ceStep == 6 {
				return times.Mean(runCtx)
			}

			perSampleLoss := times.Sum(runCtx, classDim)
			if *ceStep == 7 {
				return perSampleLoss.Mean(runCtx)
			}

			meanLoss := perSampleLoss.Mean(runCtx)
			if *ceStep == 8 {
				return meanLoss
			}

			if *ceStep == 9 {
				return meanLoss.Negate(runCtx)
			}

			panic("invalid ce_step")
		}

		if *forwardLevel == 3 || *forwardLevel == 4 {
			if *ceContext == "run" {
				return crossEntropy(runCtx, yOneHot, logits)
			}
			if *ceContext != "root" {
				panic("invalid ce_context")
			}
			return crossEntropy(ctx, yOneHot, logits)
		}
		panic("invalid forward_level")
	}

	for i := range *epochs {
		epochCtx := ctx.Epoch(i)
		before := ctx.NumAllocatedBlocks()
		beforeAllocated, capacity, beforeBlocks, beforeFreeBlocks := memoryStats(ctx)

		var ix shapes.Tensor
		var emb shapes.Tensor
		var yBatch shapes.Tensor
		var lossValue shapes.Tensor
		var graph shapes.ComputationGraph

		if *stage >= 1 {
			ix = shapes.FloatRandom(epochCtx, shapes.Shape{32}, 0, float32(X.Shape()[0])).I64(epochCtx)
			emb = embLayer.Forward(epochCtx, X.Get(epochCtx, ix))
			yBatch = Y.Get(epochCtx, ix)
		}

		if *stage >= 2 {
			lossValue = forward(epochCtx, emb, yBatch)
		}

		if *stage >= 3 {
			_ = lossValue.Get(epochCtx, 0).Item()
		}

		if *stage >= 4 {
			graph = lossValue.Backward(epochCtx)
		}

		if *stage >= 5 {
			sgd(graph)
			optimizer.ZeroGrad(ctx, graph)
		}

		epochCtx.Finish()
		after := ctx.NumAllocatedBlocks()
		afterAllocated, _, afterBlocks, afterFreeBlocks := memoryStats(ctx)

		if *printEvery <= 1 || i%*printEvery == 0 || i == *epochs-1 {
			fmt.Printf(
				"[stage %d fwd %d ce %d cectx %s epoch %04d] usedBlocks=%d->%d (%+d) blocks=%d->%d free=%d->%d allocated=%d->%d (%+d) capacity=%d\n",
				*stage,
				*forwardLevel,
				*ceStep,
				*ceContext,
				i,
				before,
				after,
				after-before,
				beforeBlocks,
				afterBlocks,
				beforeFreeBlocks,
				afterFreeBlocks,
				beforeAllocated,
				afterAllocated,
				int64(afterAllocated)-int64(beforeAllocated),
				capacity,
			)
		}
	}
}
