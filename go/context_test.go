package shapes

import (
	stdctx "context"
	"testing"
)

func mustPanic(t *testing.T, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	fn()
}

func TestNew(t *testing.T) {
	ctx := New(stdctx.Background())
	defer ctx.Finish()

	if ctx == nil {
		t.Fatal("New() returned nil")
	}
	if ctx.UnsafePtr() == nil {
		t.Fatal("context has nil C pointer")
	}
	if ctx.UnsafeMemory() == nil {
		t.Fatal("context has nil memory pointer")
	}
}

func TestWithGrad(t *testing.T) {
	ctx := New(stdctx.Background())
	defer ctx.Finish()
	if ctx.GradEnabled() {
		t.Error("grad should be disabled by default")
	}
	if ctx.BackwardEnabled() {
		t.Error("backward should be disabled by default")
	}

	ctx2 := New(stdctx.Background(), WithGrad(true))
	defer ctx2.Finish()
	if !ctx2.GradEnabled() {
		t.Error("grad should be enabled with WithGrad(true)")
	}
	if !ctx2.BackwardEnabled() {
		t.Error("backward should be enabled with WithGrad(true)")
	}
}

func TestFusedRequiresGrad(t *testing.T) {
	ctx := New(stdctx.Background())
	defer ctx.Finish()

	mustPanic(t, func() {
		_ = ctx.Fused()
	})

	ctxGrad := New(stdctx.Background(), WithGrad(true))
	defer ctxGrad.Finish()
	fused := ctxGrad.Fused()
	defer fused.Finish()

	if !fused.GradEnabled() {
		t.Fatal("fused context should keep grad enabled")
	}
	if fused.BackwardEnabled() {
		t.Fatal("fused context should disable backward")
	}
}

func TestNoGradRequiresGradEnabledParent(t *testing.T) {
	ctx := New(stdctx.Background())
	defer ctx.Finish()
	mustPanic(t, func() {
		_ = ctx.NoGrad()
	})

	ctxGrad := New(stdctx.Background(), WithGrad(true))
	defer ctxGrad.Finish()
	noGrad := ctxGrad.NoGrad()
	defer noGrad.Finish()

	if noGrad.GradEnabled() {
		t.Fatal("NoGrad should disable grad")
	}
}

func TestNoGraphDisablesGradAndBackward(t *testing.T) {
	ctx := New(stdctx.Background(), WithGrad(true))
	defer ctx.Finish()

	ng := ctx.NoGraph()
	defer ng.Finish()

	if ng.GradEnabled() {
		t.Fatal("NoGraph should disable grad")
	}
	if ng.BackwardEnabled() {
		t.Fatal("NoGraph should disable backward")
	}
}
