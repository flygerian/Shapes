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

func TestTrainingSetsIsTraining(t *testing.T) {
	ctx := New(stdctx.Background(), WithGrad(true))
	defer ctx.Finish()

	if ctx.IsTraining() {
		t.Fatal("new context should not be training")
	}

	ctx.Training(1)
	if !ctx.IsTraining() {
		t.Fatal("Training() should set context to training mode")
	}
}

func TestInferenceClearsIsTraining(t *testing.T) {
	ctx := New(stdctx.Background(), WithGrad(true))
	defer ctx.Finish()

	ctx.Training(1)
	ctx.Inference()

	if ctx.IsTraining() {
		t.Fatal("Inference() should clear training mode")
	}
}

func TestDerivedContextsCarryTrainingState(t *testing.T) {
	ctx := New(stdctx.Background(), WithGrad(true))
	defer ctx.Finish()

	ctx.Training(1)

	fused := ctx.Fused()
	defer fused.Finish()
	if !fused.IsTraining() {
		t.Fatal("derived context should inherit training mode")
	}

	ctx.Inference()
	noGraph := ctx.NoGraph()
	defer noGraph.Finish()
	if noGraph.IsTraining() {
		t.Fatal("derived context should inherit inference mode")
	}
}

func TestBackwardDisabledDisablesBackward(t *testing.T) {
	ctx := New(stdctx.Background(), WithGrad(true))
	defer ctx.Finish()

	disabled := ctx.BackwardDisabled()
	defer disabled.Finish()

	if !disabled.GradEnabled() {
		t.Fatal("BackwardDisabled should keep grad enabled")
	}
	if disabled.BackwardEnabled() {
		t.Fatal("BackwardDisabled should disable backward")
	}

	sub, ok := disabled.(*subContext)
	if !ok {
		t.Fatal("expected BackwardDisabled to return *subContext")
	}
	if sub.fused {
		t.Fatal("BackwardDisabled should not mark the subcontext as fused")
	}
}

func TestFusedPreservesParentFlagsAndMarksFlushBoundary(t *testing.T) {
	ctx := New(stdctx.Background(), WithGrad(true))
	defer ctx.Finish()

	fused := ctx.Fused()
	defer fused.Finish()

	if !fused.GradEnabled() {
		t.Fatal("Fused should preserve grad tracking")
	}
	if !fused.BackwardEnabled() {
		t.Fatal("Fused should preserve backward behavior")
	}

	sub, ok := fused.(*subContext)
	if !ok {
		t.Fatal("expected Fused to return *subContext")
	}
	if !sub.fused {
		t.Fatal("Fused should mark the subcontext as fused")
	}
}

func TestNonFusedContextWithoutFusedAncestorFlushesOnFinish(t *testing.T) {
	ctx := New(stdctx.Background(), WithGrad(true))
	defer ctx.Finish()

	sc, ok := ctx.NoGraph().(*subContext)
	if !ok {
		t.Fatal("expected NoGraph to return *subContext")
	}
	defer sc.Finish()

	if !sc.shouldFlushOnFinish() {
		t.Fatal("expected non-fused context without fused ancestor to flush")
	}
}

func TestNonFusedContextUnderFusedAncestorDefersFlush(t *testing.T) {
	ctx := New(stdctx.Background(), WithGrad(true))
	defer ctx.Finish()

	fused, ok := ctx.Fused().(*subContext)
	if !ok {
		t.Fatal("expected Fused to return *subContext")
	}
	defer fused.Finish()

	child, ok := fused.NoGraph().(*subContext)
	if !ok {
		t.Fatal("expected NoGraph child to return *subContext")
	}
	defer child.Finish()

	if !child.hasFusedAncestor() {
		t.Fatal("expected child context to detect fused ancestor")
	}
	if child.shouldFlushOnFinish() {
		t.Fatal("expected child context under fused ancestor to defer flush")
	}
}

func TestNestedFusedContextStillFlushesOnFinish(t *testing.T) {
	ctx := New(stdctx.Background(), WithGrad(true))
	defer ctx.Finish()

	parent, ok := ctx.Fused().(*subContext)
	if !ok {
		t.Fatal("expected Fused to return *subContext")
	}
	defer parent.Finish()

	child, ok := parent.Fused().(*subContext)
	if !ok {
		t.Fatal("expected nested Fused to return *subContext")
	}
	defer child.Finish()

	if !child.hasFusedAncestor() {
		t.Fatal("expected nested fused context to detect fused ancestor")
	}
	if !child.shouldFlushOnFinish() {
		t.Fatal("expected fused context to flush even with fused ancestor")
	}
}

func TestFusedStepContextPreservesStepSemantics(t *testing.T) {
	ctx := New(stdctx.Background(), WithGrad(true))
	defer ctx.Finish()

	trainingCtx := ctx.Training(1)
	epochCtx := trainingCtx.Epoch(1)
	defer epochCtx.Finish()

	stepCtx, ok := epochCtx.Step().Fused().(StepContext)
	if !ok {
		t.Fatal("expected fused step to still satisfy StepContext")
	}
	defer stepCtx.Finish()

	sub, ok := stepCtx.(*subContext)
	if !ok {
		t.Fatal("expected fused step to be backed by *subContext")
	}
	if sub.subContextType != SubContextTypeStep {
		t.Fatalf("expected fused step subcontext type %d, got %d", SubContextTypeStep, sub.subContextType)
	}
	if !sub.sweepAfterFinish {
		t.Fatal("expected fused step to preserve step sweep behavior")
	}

	stepCtx.SetStepLoss(0.25)
}

func TestFusedContextPreservesParentSweepBehavior(t *testing.T) {
	ctx := New(stdctx.Background(), WithGrad(true))
	defer ctx.Finish()

	trainingCtx := ctx.Training(1)
	epochCtx := trainingCtx.Epoch(1)
	defer epochCtx.Finish()

	stepCtx, ok := epochCtx.Step().(*subContext)
	if !ok {
		t.Fatal("expected step context to be backed by *subContext")
	}
	if !stepCtx.sweepAfterFinish {
		t.Fatal("expected plain step context to sweep after finish")
	}

	fusedStep, ok := stepCtx.Fused().(*subContext)
	if !ok {
		t.Fatal("expected fused step to be backed by *subContext")
	}
	defer fusedStep.Finish()

	if !fusedStep.sweepAfterFinish {
		t.Fatal("expected fused step to preserve parent sweep behavior")
	}
}

func TestEpochStepRecordsStepLossSeparately(t *testing.T) {
	ctx := New(stdctx.Background(), WithGrad(true))
	defer ctx.Finish()

	trainingCtx := ctx.Training(2)
	stats := trainingCtx.TrainingStats()

	epochCtx := trainingCtx.Epoch(1)
	stepCtx := epochCtx.Step()
	stepCtx.SetStepLoss(0.25)
	stepCtx.Finish()

	epochCtx.SetLoss(0.5)
	epochCtx.Finish()

	if stats.Step != 1 {
		t.Fatalf("expected current step to be 1, got %d", stats.Step)
	}
	if stats.StepLoss != 0.25 {
		t.Fatalf("expected step loss 0.25, got %f", stats.StepLoss)
	}
	if len(stats.StepLossHistoryX) != 1 || stats.StepLossHistoryX[0] != 1 {
		t.Fatalf("unexpected step loss x history: %#v", stats.StepLossHistoryX)
	}
	if len(stats.StepLossHistory) != 1 || stats.StepLossHistory[0] != 250 {
		t.Fatalf("unexpected step loss history: %#v", stats.StepLossHistory)
	}
	if stats.Loss != 0.5 {
		t.Fatalf("expected epoch loss 0.5, got %f", stats.Loss)
	}
	if len(stats.LossHistoryX) != 1 || stats.LossHistoryX[0] != 1 {
		t.Fatalf("unexpected epoch loss x history: %#v", stats.LossHistoryX)
	}
	if len(stats.LossHistory) != 1 || stats.LossHistory[0] != 500 {
		t.Fatalf("unexpected epoch loss history: %#v", stats.LossHistory)
	}
}

func TestStepRequiresEpochContext(t *testing.T) {
	ctx := New(stdctx.Background(), WithGrad(true))
	defer ctx.Finish()

	trainingCtx := ctx.Training(1)

	mustPanic(t, func() {
		_ = step(trainingCtx)
	})
}

func TestWithNumStepsSetsTrainingStats(t *testing.T) {
	ctx := New(stdctx.Background(), WithGrad(true))
	defer ctx.Finish()

	trainingCtx := ctx.Training(3, WithNumSteps(42))

	if got := trainingCtx.TrainingStats().NumSteps; got != 42 {
		t.Fatalf("NumSteps = %d, want 42", got)
	}
}
