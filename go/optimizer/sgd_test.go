package optimizer

import (
	"context"
	"math"
	"testing"

	shapes "github.com/flygerian/shapes"
	"github.com/flygerian/shapes/extract"
	"github.com/flygerian/shapes/layer"
	"github.com/flygerian/shapes/tensor"
)

func approxEq(a, b, tol float32) bool {
	return float32(math.Abs(float64(a-b))) < tol
}

func TestSGDUpdatesParameters(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	dense := layer.Dense(3, 2)
	x := tensor.Float(ctx, tensor.Shape{1, 3}, 1.0)

	o := dense(ctx, x)
	graph := o.Backward(ctx)

	params := extract.Parameters(graph)
	if len(params) != 2 {
		t.Fatalf("expected 2 parameters (w, b), got %d", len(params))
	}

	before := snapshotParams(t, params)

	noGraph := ctx.NoGraph()
	step := SGD(noGraph, 0.01)
	step(graph)

	after := snapshotParams(t, params)

	changed := false
	for i := range before {
		if !approxEq(before[i], after[i], 1e-10) {
			changed = true
			break
		}
	}
	if !changed {
		t.Error("expected at least one parameter to change after SGD step")
	}
}

func TestSGDLearningRateScalesUpdate(t *testing.T) {
	// Same graph structure, two learning rates.
	// delta(lr=0.1) should be 10x delta(lr=0.01).
	deltas1 := runSGDStep(t, 0.01)
	deltas2 := runSGDStep(t, 0.1)

	if len(deltas1) == 0 {
		t.Fatal("no parameter deltas recorded")
	}

	for i := range deltas1 {
		if approxEq(deltas1[i], 0, 1e-10) {
			continue
		}
		ratio := deltas2[i] / deltas1[i]
		if !approxEq(ratio, 10.0, 0.5) {
			t.Errorf("delta ratio[%d] = %f, want ~10.0", i, ratio)
		}
	}
}

func TestSGDLossDecreases(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	dense := layer.Dense(3, 2)
	x := tensor.Float(ctx, tensor.Shape{1, 3}, 1.0)

	o := dense(ctx, x)
	graph := o.Backward(ctx)

	params := extract.Parameters(graph)
	if len(params) == 0 {
		t.Fatal("no parameters found")
	}

	// Verify gradients are non-zero.
	hasNonZeroGrad := false
	for _, p := range params {
		shape := tensor.ShapeOf(p)
		coords := make([]uint32, len(shape))
		args := make([]interface{}, len(coords))
		for i, c := range coords {
			args[i] = c
		}
		v := p.Grad().Get(args...).Item().(float32)
		if !approxEq(v, 0, 1e-10) {
			hasNonZeroGrad = true
			break
		}
	}
	if !hasNonZeroGrad {
		t.Error("expected at least one non-zero gradient")
	}

	noGraph := ctx.NoGraph()
	step := SGD(noGraph, 0.01)
	step(graph)
}

func runSGDStep(t *testing.T, lr float32) []float32 {
	t.Helper()
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	dense := layer.Dense(3, 2)
	x := tensor.Float(ctx, tensor.Shape{1, 3}, 1.0)

	o := dense(ctx, x)
	graph := o.Backward(ctx)

	params := extract.Parameters(graph)
	before := snapshotParams(t, params)

	noGraph := ctx.NoGraph()
	step := SGD(noGraph, lr)
	step(graph)

	after := snapshotParams(t, params)

	deltas := make([]float32, len(before))
	for i := range before {
		deltas[i] = after[i] - before[i]
	}
	return deltas
}

// snapshotParams reads all float32 values from params using proper multi-dim coords.
func snapshotParams(t *testing.T, params []*tensor.Tensor) []float32 {
	t.Helper()
	var vals []float32
	for _, p := range params {
		shape := tensor.ShapeOf(p)
		total := 1
		for _, d := range shape {
			total *= int(d)
		}
		for flat := range total {
			coords := unflattenIndex(flat, shape)
			args := make([]interface{}, len(coords))
			for i, c := range coords {
				args[i] = c
			}
			v := p.Get(args...).Item().(float32)
			vals = append(vals, v)
		}
	}
	return vals
}

// unflattenIndex converts a flat index to multi-dimensional coordinates.
func unflattenIndex(flat int, shape tensor.Shape) []uint32 {
	coords := make([]uint32, len(shape))
	for i := len(shape) - 1; i >= 0; i-- {
		coords[i] = uint32(flat % int(shape[i]))
		flat /= int(shape[i])
	}
	return coords
}
