package layer

import (
	"context"
	"math"
	"testing"

	shapes "github.com/flygerian/shapes"
)

func TestBatchNormReturnsResult(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	bn := BatchNorm(ctx, 3)
	x := shapes.Float(ctx, shapes.Shape{2, 3}, 1.0)

	o := bn.Forward(ctx, x)
	if o == nil {
		t.Fatal("BatchNorm returned nil")
	}

	shape := o.Shape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Fatalf("expected shape [2,3], got %v", shape)
	}

	if o.Op() != shapes.OpBatchNorm {
		t.Fatalf("expected OpBatchNorm, got %s", o.Op())
	}
}

func TestBatchNorm1DInput(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	bn := BatchNorm(ctx, 4)
	x := shapes.Float(ctx, shapes.Shape{4}, 2.0)

	o := bn.Forward(ctx, x)
	shape := o.Shape()
	if len(shape) != 1 || shape[0] != 4 {
		t.Fatalf("expected shape [4], got %v", shape)
	}
}

func TestBatchNormNormalizesPerFeature(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	bn := BatchNorm(ctx, 2)
	x := shapes.FromFloat32(ctx, shapes.Shape{2, 2}, []float32{
		1.0, 2.0,
		3.0, 4.0,
	})

	o := bn.Forward(ctx, x)

	wantMag := float32(1.0 / math.Sqrt(1.0+defaultBatchNormEpsilon))
	tests := []struct {
		i, j uint32
		want float32
	}{
		{0, 0, -wantMag},
		{0, 1, -wantMag},
		{1, 0, wantMag},
		{1, 1, wantMag},
	}

	for _, tt := range tests {
		got := o.Get(ctx, tt.i, tt.j).Item().(float32)
		if math.Abs(float64(got-tt.want)) > 1e-4 {
			t.Fatalf("BatchNorm[%d,%d] = %f, want %f", tt.i, tt.j, got, tt.want)
		}
	}
}

func TestBatchNormBackward(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	bn := BatchNorm(ctx, 3)
	x := shapes.Float(ctx, shapes.Shape{2, 3}, 1.0)

	o := bn.Forward(ctx, x)
	o.Backward(ctx)

	hidden := o.HiddenState()
	if len(hidden) != 2 {
		t.Fatalf("expected 2 hidden tensors (gamma,beta), got %d", len(hidden))
	}

	gammaGrad := hidden[0].Grad()
	betaGrad := hidden[1].Grad()
	xGrad := o.Inputs()[0].Grad()

	gammaShape := gammaGrad.Shape()
	if len(gammaShape) != 1 || gammaShape[0] != 3 {
		t.Fatalf("expected gamma grad shape [3], got %v", gammaShape)
	}

	betaShape := betaGrad.Shape()
	if len(betaShape) != 1 || betaShape[0] != 3 {
		t.Fatalf("expected beta grad shape [3], got %v", betaShape)
	}

	xShape := xGrad.Shape()
	if len(xShape) != 2 || xShape[0] != 2 || xShape[1] != 3 {
		t.Fatalf("expected x grad shape [2,3], got %v", xShape)
	}
}

func TestBatchNormPanicsOnMismatchedFeatureSize(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	bn := BatchNorm(ctx, 3)
	x := shapes.Float(ctx, shapes.Shape{2, 4}, 1.0)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for mismatched feature size, got nil")
		}
	}()

	bn.Forward(ctx, x)
}

func TestBatchNormTracksRunningStatsDuringTraining(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	ctx.Training(2)
	bnLayer := BatchNorm(ctx, 1)
	bn := bnLayer.(*batchNorm)

	x1 := shapes.FromFloat32(ctx, shapes.Shape{2, 1}, []float32{0.0, 2.0}) // mean=1, var=1
	e1 := ctx.Epoch(1)
	_ = bn.Forward(e1, x1)
	e1.Finish()

	if !bn.runningStatsInitialized {
		t.Fatal("expected running stats to initialize in training mode")
	}
	if got := bn.runningMean[0]; math.Abs(float64(got-1.0)) > 1e-5 {
		t.Fatalf("running mean after first batch = %f, want 1.0", got)
	}
	if got := bn.runningVar[0]; math.Abs(float64(got-1.0)) > 1e-5 {
		t.Fatalf("running var after first batch = %f, want 1.0", got)
	}

	x2 := shapes.FromFloat32(ctx, shapes.Shape{2, 1}, []float32{10.0, 14.0}) // mean=12, var=4
	e2 := ctx.Epoch(2)
	_ = bn.Forward(e2, x2)
	e2.Finish()

	wantMean := float32(1.0*(1.0-defaultBatchNormMomentum) + 12.0*defaultBatchNormMomentum)
	wantVar := float32(1.0*(1.0-defaultBatchNormMomentum) + 4.0*defaultBatchNormMomentum)
	if got := bn.runningMean[0]; math.Abs(float64(got-wantMean)) > 1e-5 {
		t.Fatalf("running mean after second batch = %f, want %f", got, wantMean)
	}
	if got := bn.runningVar[0]; math.Abs(float64(got-wantVar)) > 1e-5 {
		t.Fatalf("running var after second batch = %f, want %f", got, wantVar)
	}
}

func TestBatchNormInferenceUsesRunningStats(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	ctx.Training(1)
	bnLayer := BatchNorm(ctx, 1)

	trainX := shapes.FromFloat32(ctx, shapes.Shape{2, 1}, []float32{0.0, 2.0}) // mean=1, var=1
	trainEpoch := ctx.Epoch(1)
	_ = bnLayer.Forward(trainEpoch, trainX)
	trainEpoch.Finish()

	ctx.Inference()
	inferX := shapes.FromFloat32(ctx, shapes.Shape{2, 1}, []float32{100.0, 100.0})
	out := bnLayer.Forward(ctx, inferX)

	want := float32((100.0 - 1.0) / math.Sqrt(1.0+defaultBatchNormEpsilon))
	for i := range uint32(2) {
		got := out.Get(ctx, i, 0).Item().(float32)
		if math.Abs(float64(got-want)) > 1e-3 {
			t.Fatalf("inference output[%d,0] = %f, want %f", i, got, want)
		}
	}
}
