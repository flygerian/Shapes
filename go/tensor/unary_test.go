package tensor

import (
	"context"
	"math"
	"testing"

	shapes "github.com/flygerian/shapes"
)

func TestPow(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	a := Float(ctx, Shape{2, 2}, 3.0)
	result := a.Pow(2.0)

	for i := range uint32(2) {
		for j := range uint32(2) {
			got, err := result.GetF32(i, j)
			if err != nil {
				t.Fatalf("GetF32(%d,%d): %v", i, j, err)
			}
			if math.Abs(float64(got)-9.0) > 1e-4 {
				t.Errorf("Pow[%d,%d] = %f, want 9.0", i, j, got)
			}
		}
	}
}

func TestPowFractional(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	a := Float(ctx, Shape{2}, 4.0)
	result := a.Pow(0.5)

	for i := range uint32(2) {
		got, err := result.GetF32(i)
		if err != nil {
			t.Fatalf("GetF32(%d): %v", i, err)
		}
		if math.Abs(float64(got)-2.0) > 1e-4 {
			t.Errorf("Pow(0.5)[%d] = %f, want 2.0", i, got)
		}
	}
}

func TestExp(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	a := Float(ctx, Shape{2, 2}, 1.0)
	result := a.Exp()

	want := float32(math.E)
	for i := range uint32(2) {
		for j := range uint32(2) {
			got, err := result.GetF32(i, j)
			if err != nil {
				t.Fatalf("GetF32(%d,%d): %v", i, j, err)
			}
			if math.Abs(float64(got-want)) > 1e-4 {
				t.Errorf("Exp[%d,%d] = %f, want %f", i, j, got, want)
			}
		}
	}
}

func TestExpZero(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	a := Float(ctx, Shape{3}, 0.0)
	result := a.Exp()

	for i := range uint32(3) {
		got, err := result.GetF32(i)
		if err != nil {
			t.Fatalf("GetF32(%d): %v", i, err)
		}
		if math.Abs(float64(got)-1.0) > 1e-4 {
			t.Errorf("Exp(0)[%d] = %f, want 1.0", i, got)
		}
	}
}
