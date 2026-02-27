package loss

import (
	"context"
	"math"
	"testing"

	shapes "github.com/flygerian/shapes"
)

// logitsForProb returns logits such that softmax(logits)[targetClass] ≈ targetProb.
// All non-target classes receive equal probability sharing the remainder.
// For a 4-class case with targetProb=0.7: logits = [0, log(7), 0, 0]
// because softmax normalises exp([0,log(7),0,0]) = [1,7,1,1] → [0.1, 0.7, 0.1, 0.1].

func TestCrossEntropy(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Finish()

	// One-hot encoded: class 1 is the true label
	yGround := ctx.FromFloat32(shapes.Shape{4}, []float32{0.0, 1.0, 0.0, 0.0})

	// Logits: [0, log(7), 0, 0] → softmax → [0.1, 0.7, 0.1, 0.1]
	yLogits := ctx.FromFloat32(shapes.Shape{4}, []float32{0.0, float32(math.Log(7)), 0.0, 0.0})

	loss := CrossEntropy(yGround, yLogits)

	// Cross entropy = -log(0.7)
	got := loss.Tensor().Get(ctx, 0).Item().(float32)
	want := float32(-math.Log(0.7))
	if !approxEq(got, want, 1e-4) {
		t.Errorf("CrossEntropy = %f, want %f", got, want)
	}
}

func TestCrossEntropyPerfectPrediction(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Finish()

	// Perfect prediction: logit for true class (class 1) is large, others are very small.
	// softmax([0, 10, 0]) ≈ [0, 1, 0] → loss ≈ -log(1) ≈ 0.
	yGround := ctx.FromFloat32(shapes.Shape{3}, []float32{0.0, 1.0, 0.0})
	yLogits := ctx.FromFloat32(shapes.Shape{3}, []float32{0.0, 10.0, 0.0})

	loss := CrossEntropy(yGround, yLogits)

	got := loss.Tensor().Get(ctx, 0).Item().(float32)
	if got > 1e-3 {
		t.Errorf("CrossEntropy = %f, want approximately 0.0", got)
	}
}

func TestCrossEntropy2D(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Finish()

	// Batch of 2 samples, 3 classes each.
	// Sample 1: true class 0 — logits [log(8), log(1), log(1)] → softmax [0.8, 0.1, 0.1]
	// Sample 2: true class 2 — logits [log(1), log(1), log(8)] → softmax [0.1, 0.1, 0.8]
	yGround := ctx.FromFloat32(shapes.Shape{2, 3}, []float32{
		1.0, 0.0, 0.0,
		0.0, 0.0, 1.0,
	})
	l8 := float32(math.Log(8))
	yLogits := ctx.FromFloat32(shapes.Shape{2, 3}, []float32{
		l8, 0.0, 0.0,
		0.0, 0.0, l8,
	})

	loss := CrossEntropy(yGround, yLogits)

	// Expected: mean(-log(0.8), -log(0.8)) = -log(0.8)
	got := loss.Tensor().Get(ctx, 0).Item().(float32)
	want := float32(-math.Log(0.8))
	if !approxEq(got, want, 1e-4) {
		t.Errorf("CrossEntropy = %f, want %f", got, want)
	}
}

func TestCrossEntropyBackward(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	// 2-class case, single sample (1D).
	// yGround = [1, 0] (true class is 0)
	// logits = [0, 0] → softmax = [0.5, 0.5]
	// ∂loss/∂logits = probs - yGround = [0.5-1, 0.5-0] = [-0.5, 0.5]
	// (batch_size=1, so no division by batch)
	yGround := ctx.FromFloat32(shapes.Shape{2}, []float32{1.0, 0.0})
	yLogits := ctx.FromFloat32(shapes.Shape{2}, []float32{0.0, 0.0})

	loss := CrossEntropy(yGround, yLogits)
	loss.Tensor().Backward(ctx)

	logitsGrad := yLogits.Tensor().Grad()
	if logitsGrad == nil {
		t.Fatal("expected gradient on logits")
	}

	got0 := logitsGrad.(*shapes.Tensor).Get(ctx, 0).Item().(float32)
	got1 := logitsGrad.(*shapes.Tensor).Get(ctx, 1).Item().(float32)
	if !approxEq(got0, -0.5, 1e-4) {
		t.Errorf("d(loss)/d(logits[0]) = %f, want -0.5", got0)
	}
	if !approxEq(got1, 0.5, 1e-4) {
		t.Errorf("d(loss)/d(logits[1]) = %f, want 0.5", got1)
	}
}

func TestCrossEntropyIncorrectPrediction(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Finish()

	// True label: class 0
	// Logits: [log(1), log(8), log(1)] → softmax ≈ [0.1, 0.8, 0.1]
	// loss = -log(0.1)
	yGround := ctx.FromFloat32(shapes.Shape{3}, []float32{1.0, 0.0, 0.0})
	l8 := float32(math.Log(8))
	yLogits := ctx.FromFloat32(shapes.Shape{3}, []float32{0.0, l8, 0.0})

	loss := CrossEntropy(yGround, yLogits)

	got := loss.Tensor().Get(ctx, 0).Item().(float32)
	want := float32(-math.Log(0.1))
	if !approxEq(got, want, 1e-4) {
		t.Errorf("CrossEntropy = %f, want %f", got, want)
	}
}
