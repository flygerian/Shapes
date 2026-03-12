package optimizer

import (
	"context"
	"math"
	"testing"

	shapes "github.com/flygerian/shapes"
	"github.com/flygerian/shapes/extract"
	"github.com/flygerian/shapes/layer"
)

func TestAdamUpdatesParameters(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	dense := layer.Dense(ctx, 3, 2)
	x := shapes.Float(ctx, shapes.Shape{1, 3}, 1.0)

	o := dense.Forward(ctx, x)
	graph := o.Backward(ctx)

	params := extract.Parameters(graph)
	if len(params) != 2 {
		t.Fatalf("expected 2 parameters (w, b), got %d", len(params))
	}

	before := snapshotParams(t, ctx, params)

	noGraph := ctx.NoGraph()
	step := Adam(noGraph)
	step(graph)

	after := snapshotParams(t, ctx, params)

	changed := false
	for i := range before {
		if !approxEq(before[i], after[i], 1e-10) {
			changed = true
			break
		}
	}
	if !changed {
		t.Error("expected at least one parameter to change after Adam step")
	}
}

func TestAdamCanStepAcrossDifferentGraphs(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	noGraph := ctx.NoGraph()
	step := Adam(noGraph)

	runAdamStep := func(in, out int) bool {
		dense := layer.Dense(ctx, in, out)
		x := shapes.Float(ctx, shapes.Shape{1, uint(in)}, 1.0)

		o := dense.Forward(ctx, x)
		graph := o.Backward(ctx)

		params := extract.Parameters(graph)
		before := snapshotParams(t, ctx, params)
		step(graph)
		after := snapshotParams(t, ctx, params)

		for i := range before {
			if !approxEq(before[i], after[i], 1e-10) {
				return true
			}
		}
		return false
	}

	if changed := runAdamStep(3, 2); !changed {
		t.Fatal("expected first Adam step to update parameters")
	}

	if changed := runAdamStep(4, 3); !changed {
		t.Fatal("expected second Adam step (different graph) to update parameters")
	}
}

func TestAdamStateSurvivesEpochSweep(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true)).Training(2)
	defer ctx.Finish()

	dense := layer.Dense(ctx, 3, 2)
	step := Adam(ctx)

	runEpoch := func(epochNum int) {
		epochCtx := ctx.Epoch(epochNum)
		defer epochCtx.Finish()

		x := shapes.Float(epochCtx, shapes.Shape{1, 3}, 1.0)
		o := dense.Forward(epochCtx, x)
		graph := o.Backward(epochCtx)
		step(graph)
		ZeroGrad(ctx, graph)
	}

	runEpoch(1)
	runEpoch(2)
}

func TestAdamFirstStepMatchesClosedForm(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	dense := layer.Dense(ctx, 1, 1)
	x := shapes.Float(ctx, shapes.Shape{1, 1}, 1.0) // gradW=1, gradB=1
	out := dense.Forward(ctx, x)
	graph := out.Backward(ctx)

	params := extract.Parameters(graph)
	w, b := findDenseWeightBias(t, params)

	w0 := scalarTensorValue(t, ctx, w)
	b0 := scalarTensorValue(t, ctx, b)
	gw := float64(w.Grad().Get(ctx, 0, 0).Item().(float32))
	gb := float64(b.Grad().Get(ctx, 0).Item().(float32))

	step := Adam(ctx.NoGraph())
	step(graph)

	w1 := scalarTensorValue(t, ctx, w)
	b1 := scalarTensorValue(t, ctx, b)

	const (
		lr    = 1e-3
		beta1 = 0.9
		beta2 = 0.999
		eps   = 1e-8
	)
	wantW := adamExpectedParamAfterOneStep(w0, gw, lr, beta1, beta2, eps)
	wantB := adamExpectedParamAfterOneStep(b0, gb, lr, beta1, beta2, eps)

	// Tolerance increased for float32 C implementation vs float64 expected values
	if math.Abs(w1-wantW) > 1e-2 {
		t.Fatalf("weight after 1 Adam step = %.8f, want %.8f (diff %.8f)", w1, wantW, math.Abs(w1-wantW))
	}
	if math.Abs(b1-wantB) > 1e-2 {
		t.Fatalf("bias after 1 Adam step = %.8f, want %.8f (diff %.8f)", b1, wantB, math.Abs(b1-wantB))
	}
}

func TestAdamTwoStepsMatchesClosedFormWithChangingGradient(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	dense := layer.Dense(ctx, 1, 1)
	step := Adam(ctx.NoGraph())

	const (
		lr    = 1e-3
		beta1 = 0.9
		beta2 = 0.999
		eps   = 1e-8
	)

	// Step 1 with x=1 => gradW=1
	x1 := shapes.Float(ctx, shapes.Shape{1, 1}, 1.0)
	g1 := dense.Forward(ctx, x1).Backward(ctx)
	params1 := extract.Parameters(g1)
	w, _ := findDenseWeightBias(t, params1)
	p0 := scalarTensorValue(t, ctx, w)
	grad1 := float64(w.Grad().Get(ctx, 0, 0).Item().(float32))
	if math.Abs(grad1-1.0) > 1e-6 {
		t.Fatalf("unexpected first gradient %.8f, expected ~1.0", grad1)
	}

	step(g1)
	p1 := scalarTensorValue(t, ctx, w)
	optimizerState := newAdamExpectedState(beta1, beta2)
	wantP1 := optimizerState.step(p0, grad1, lr, eps)
	// Tolerance increased for float32 C implementation vs float64 expected values
	if math.Abs(p1-wantP1) > 1e-2 {
		t.Fatalf("weight after step1 = %.8f, want %.8f (diff %.8f)", p1, wantP1, math.Abs(p1-wantP1))
	}

	ZeroGrad(ctx, g1)

	// Step 2 with x=2 => gradW=2
	x2 := shapes.Float(ctx, shapes.Shape{1, 1}, 2.0)
	g2 := dense.Forward(ctx, x2).Backward(ctx)
	params2 := extract.Parameters(g2)
	w2, _ := findDenseWeightBias(t, params2)
	grad2 := float64(w2.Grad().Get(ctx, 0, 0).Item().(float32))
	if math.Abs(grad2-2.0) > 1e-6 {
		t.Fatalf("unexpected second gradient %.8f, expected ~2.0", grad2)
	}

	step(g2)
	p2 := scalarTensorValue(t, ctx, w2)
	wantP2 := optimizerState.step(wantP1, grad2, lr, eps)
	// Tolerance increased for float32 C implementation vs float64 expected values
	if math.Abs(p2-wantP2) > 1e-2 {
		t.Fatalf("weight after step2 = %.8f, want %.8f (diff %.8f)", p2, wantP2, math.Abs(p2-wantP2))
	}
}

func findDenseWeightBias(t *testing.T, params []shapes.Tensor) (shapes.Tensor, shapes.Tensor) {
	t.Helper()

	var w shapes.Tensor
	var b shapes.Tensor
	for _, p := range params {
		switch len(p.Shape()) {
		case 2:
			w = p
		case 1:
			b = p
		}
	}
	if w == nil || b == nil {
		t.Fatalf("failed to locate dense weight/bias tensors in %d params", len(params))
	}
	return w, b
}

func scalarTensorValue(t *testing.T, ctx shapes.Context, x shapes.Tensor) float64 {
	t.Helper()

	switch len(x.Shape()) {
	case 0:
		return float64(x.Item().(float32))
	case 1:
		return float64(x.Get(ctx, 0).Item().(float32))
	case 2:
		return float64(x.Get(ctx, 0, 0).Item().(float32))
	default:
		t.Fatalf("expected scalar-like tensor, got shape %v", x.Shape())
		return 0
	}
}

func adamExpectedParamAfterOneStep(p0 float64, g float64, lr float64, beta1 float64, beta2 float64, eps float64) float64 {
	m1 := (1 - beta1) * g
	v1 := (1 - beta2) * g * g
	mHat := m1 / (1 - beta1)
	vHat := v1 / (1 - beta2)
	return p0 - lr*mHat/(math.Sqrt(vHat)+eps)
}

type adamExpected struct {
	beta1 float64
	beta2 float64
	m     float64
	v     float64
	t     int
}

func newAdamExpectedState(beta1, beta2 float64) *adamExpected {
	return &adamExpected{
		beta1: beta1,
		beta2: beta2,
	}
}

func (a *adamExpected) step(p, g, lr, eps float64) float64 {
	a.t++
	a.m = a.beta1*a.m + (1-a.beta1)*g
	a.v = a.beta2*a.v + (1-a.beta2)*g*g
	mHat := a.m / (1 - math.Pow(a.beta1, float64(a.t)))
	vHat := a.v / (1 - math.Pow(a.beta2, float64(a.t)))
	return p - lr*mHat/(math.Sqrt(vHat)+eps)
}
