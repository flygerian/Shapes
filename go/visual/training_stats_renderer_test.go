package visual

import (
	"strings"
	"testing"

	"github.com/flygerian/shapes"
)

func TestResolveTerminalAreaFallsBackToEnv(t *testing.T) {
	t.Setenv("COLUMNS", "77")
	t.Setenv("LINES", "33")

	area := resolveTerminalArea(0, 0, false)
	if area.Width != 77 {
		t.Fatalf("width = %d, want 77", area.Width)
	}
	if area.Height != 33 {
		t.Fatalf("height = %d, want 33", area.Height)
	}
}

func TestResolveTerminalAreaUsesDetectedSize(t *testing.T) {
	t.Setenv("COLUMNS", "77")
	t.Setenv("LINES", "33")

	area := resolveTerminalArea(101, 41, true)
	if area.Width != 101 {
		t.Fatalf("width = %d, want 101", area.Width)
	}
	if area.Height != 41 {
		t.Fatalf("height = %d, want 41", area.Height)
	}
}

func TestEnvIntFallsBackOnInvalidValue(t *testing.T) {
	t.Setenv("COLUMNS", "nope")

	if got := envInt("COLUMNS", 120); got != 120 {
		t.Fatalf("envInt returned %d, want 120", got)
	}
}

func TestEnvIntFallsBackWhenUnset(t *testing.T) {
	if got := envInt("SHAPES_TEST_UNSET_ENV", 90); got != 90 {
		t.Fatalf("envInt returned %d, want 90", got)
	}
}

func TestTrainingHeaderShowsStepCountWhenNumStepsKnown(t *testing.T) {
	stats := &shapes.TrainingStats{
		Epoch:            2,
		NumEpochs:        10,
		Step:             7,
		NumSteps:         50,
		Loss:             0.123456,
		ValidationLoss:   0.654321,
		Accuracy:         0.98,
		StepLoss:         0.111111,
		StepLossHistoryX: []int{7},
	}

	header := trainingHeader(stats)
	if !strings.Contains(header, "Step 7/50 Loss 0.111111") {
		t.Fatalf("header = %q, want step count", header)
	}
}

func TestTrainingHeaderWrapsStepToCurrentEpoch(t *testing.T) {
	stats := &shapes.TrainingStats{
		Epoch:            3,
		NumEpochs:        10,
		Step:             57,
		NumSteps:         25,
		Loss:             0.123456,
		ValidationLoss:   0.654321,
		Accuracy:         0.98,
		StepLoss:         0.111111,
		StepLossHistoryX: []int{57},
	}

	header := trainingHeader(stats)
	if !strings.Contains(header, "Step 7/25 Loss 0.111111") {
		t.Fatalf("header = %q, want per-epoch step count", header)
	}
}
