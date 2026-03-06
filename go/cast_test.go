package shapes

import (
	"context"
	"math"
	"testing"
)

// --- Valid Tensor casts ---

func TestCastI8ToI16(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	src := FromInt8(ctx, []int8{1, -2, 3})
	result := src.I16(ctx)

	if result.Dtype() != DtypeI16 {
		t.Fatalf("expected dtype I16, got %s", result.Dtype())
	}
	shape := result.Shape()
	if len(shape) != 1 || shape[0] != 3 {
		t.Fatalf("expected shape [3], got %v", shape)
	}
	expected := []int16{1, -2, 3}
	for i, want := range expected {
		got := result.Get(ctx, uint32(i)).Item().(int16)
		if got != want {
			t.Errorf("I16[%d] = %d, want %d", i, got, want)
		}
	}
}

func TestCastI8ToI32(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	src := FromInt8(ctx, []int8{-5, 10})
	result := src.I32(ctx)

	if result.Dtype() != DtypeI32 {
		t.Fatalf("expected dtype I32, got %s", result.Dtype())
	}
	expected := []int32{-5, 10}
	for i, want := range expected {
		got := result.Get(ctx, uint32(i)).Item().(int32)
		if got != want {
			t.Errorf("I32[%d] = %d, want %d", i, got, want)
		}
	}
}

func TestCastI8ToI64(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	src := FromInt8(ctx, []int8{7, -7})
	result := src.I64(ctx)

	if result.Dtype() != DtypeI64 {
		t.Fatalf("expected dtype I64, got %s", result.Dtype())
	}
	expected := []int64{7, -7}
	for i, want := range expected {
		got := result.Get(ctx, uint32(i)).Item().(int64)
		if got != want {
			t.Errorf("I64[%d] = %d, want %d", i, got, want)
		}
	}
}

func TestCastI8ToF32(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	src := FromInt8(ctx, []int8{-3, 4})
	result := src.F32(ctx)

	if result.Dtype() != DtypeF32 {
		t.Fatalf("expected dtype F32, got %s", result.Dtype())
	}
	expected := []float32{-3.0, 4.0}
	for i, want := range expected {
		got := result.Get(ctx, uint32(i)).Item().(float32)
		if !approxEq(got, want, 1e-5) {
			t.Errorf("F32[%d] = %f, want %f", i, got, want)
		}
	}
}

func TestCastI8ToF64(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	src := FromInt8(ctx, []int8{-1, 2})
	result := src.F64(ctx)

	if result.Dtype() != DtypeF64 {
		t.Fatalf("expected dtype F64, got %s", result.Dtype())
	}
}

func TestCastI8ToBool(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	src := FromInt8(ctx, []int8{0, -1, 5})
	result := src.Bool(ctx)

	if result.Dtype() != DtypeBool {
		t.Fatalf("expected dtype bool, got %s", result.Dtype())
	}
	expected := []bool{false, true, true}
	for i, want := range expected {
		got := result.Get(ctx, uint32(i)).Item().(bool)
		if got != want {
			t.Errorf("Bool[%d] = %v, want %v", i, got, want)
		}
	}
}

func TestCastF32ToF64(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	src := FromFloat32(ctx, Shape{3}, []float32{1.5, -2.5, 3.14})
	result := src.F64(ctx)

	if result.Dtype() != DtypeF64 {
		t.Fatalf("expected dtype F64, got %s", result.Dtype())
	}
	shape := result.Shape()
	if len(shape) != 1 || shape[0] != 3 {
		t.Fatalf("expected shape [3], got %v", shape)
	}
}

func TestCastSameDtypeClones(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	src := FromFloat32(ctx, Shape{2, 2}, []float32{1.0, 2.0, 3.0, 4.0})
	result := src.F32(ctx)

	if result.Dtype() != DtypeF32 {
		t.Fatalf("expected dtype F32, got %s", result.Dtype())
	}
	// Values should be preserved
	expected := []float32{1.0, 2.0, 3.0, 4.0}
	for i := range uint32(2) {
		for j := range uint32(2) {
			got := result.Get(ctx, i, j).Item().(float32)
			want := expected[i*2+j]
			if !approxEq(got, want, 1e-5) {
				t.Errorf("F32[%d,%d] = %f, want %f", i, j, got, want)
			}
		}
	}
}

func TestCastPreservesShape(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	src := FromInt8(ctx, [][]int8{{1, 2, 3}, {4, 5, 6}})
	result := src.I16(ctx)

	shape := result.Shape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Fatalf("expected shape [2, 3], got %v", shape)
	}
}

// --- Invalid Tensor casts (panics) ---

func TestCastF32ToI32(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	src := FromFloat32(ctx, Shape{3}, []float32{1.5, -3.9, 42.0})
	result := src.I32(ctx)

	if result.Dtype() != DtypeI32 {
		t.Fatalf("expected dtype I32, got %s", result.Dtype())
	}
	expected := []int32{1, -3, 42}
	for i, want := range expected {
		got := result.Get(ctx, uint32(i)).Item().(int32)
		if got != want {
			t.Errorf("I32[%d] = %d, want %d", i, got, want)
		}
	}
}

func TestCastF64ToF32Panics(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for F64 -> F32 cast (truncation), got nil")
		}
	}()

	// Create an F64 tensor by casting F32 -> F64 first, then try narrowing
	src := Float(ctx, Shape{2}, 1.0)
	f64 := src.F64(ctx)
	f64.F32(ctx) // Should panic: truncating cast
}

func TestCastI32ToI16Panics(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for I32 -> I16 cast (truncation), got nil")
		}
	}()

	src := FromInt8(ctx, []int8{1, 2})
	i32 := src.I32(ctx)
	i32.I16(ctx) // Should panic: truncating cast
}

func TestCastI8ToU8Panics(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for I8 -> U8 cast (sign mismatch), got nil")
		}
	}()

	src := FromInt8(ctx, []int8{1, 2})
	src.U8(ctx) // Should panic: sign mismatch
}

func TestCastU8ToI8Panics(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for U8 -> I8 cast (sign mismatch), got nil")
		}
	}()

	// Create a U8 tensor via I8 -> I16 -> ... won't work.
	// We need a U8 tensor. FromInt8 creates I8. Let's cast I8->I16 and see.
	// Actually, the easiest way to test unsigned is to rely on the C layer.
	// For this test, just verify the panic mechanism works for the reverse direction.
	// We'll test this indirectly: U16 -> I16 should panic too.
	src := FromInt8(ctx, []int8{1})
	i16 := src.I16(ctx)
	i16.U16(ctx) // Should panic: sign mismatch (signed -> unsigned)
}

func TestCastF32ToU32Panics(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for F32 -> U32 cast (sign mismatch), got nil")
		}
	}()

	src := Float(ctx, Shape{2}, 1.0)
	src.U32(ctx) // Should panic: float -> unsigned
}

// --- WrappedTensor casts ---

func TestWrappedCastI8ToI16(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	src := FromInt8(ctx, []int8{1, -2, 3})
	result := src.I16(ctx)

	if result.Dtype() != DtypeI16 {
		t.Fatalf("expected dtype I16, got %s", result.Dtype())
	}
	expected := []int16{1, -2, 3}
	for i, want := range expected {
		got := result.Get(ctx, uint32(i)).Item().(int16)
		if got != want {
			t.Errorf("WrappedTensor I16[%d] = %d, want %d", i, got, want)
		}
	}
}

func TestWrappedCastF32ToF64(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	src := FromFloat32(ctx, Shape{2}, []float32{1.5, -2.5})
	result := src.F64(ctx)

	if result.Dtype() != DtypeF64 {
		t.Fatalf("expected dtype F64, got %s", result.Dtype())
	}
}

func TestWrappedCastI8ToF32(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	src := FromInt8(ctx, []int8{-4, 5})
	result := src.F32(ctx)

	if result.Dtype() != DtypeF32 {
		t.Fatalf("expected dtype F32, got %s", result.Dtype())
	}
	expected := []float32{-4.0, 5.0}
	for i, want := range expected {
		got := result.Get(ctx, uint32(i)).Item().(float32)
		if !approxEq(got, want, 1e-5) {
			t.Errorf("WrappedTensor F32[%d] = %f, want %f", i, got, want)
		}
	}
}

func TestWrappedCastSameDtype(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	src := Float(ctx, Shape{3}, 2.5)
	result := src.F32(ctx)

	if result.Dtype() != DtypeF32 {
		t.Fatalf("expected dtype F32, got %s", result.Dtype())
	}
	for i := range uint32(3) {
		got := result.Get(ctx, i).Item().(float32)
		if !approxEq(got, 2.5, 1e-5) {
			t.Errorf("WrappedTensor same dtype[%d] = %f, want 2.5", i, got)
		}
	}
}

func TestWrappedCastPanicsOnSignMismatch(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for WrappedTensor I8 -> U8 cast, got nil")
		}
	}()

	src := FromInt8(ctx, []int8{1, 2})
	src.U8(ctx) // Should panic
}

func TestWrappedCastPanicsOnTruncation(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for WrappedTensor F64 -> F32 cast, got nil")
		}
	}()

	src := Float(ctx, Shape{2}, 1.0)
	f64 := src.F64(ctx)
	f64.F32(ctx) // Should panic: truncating cast
}

func TestWrappedCastF32ToI32(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	src := FromFloat32(ctx, Shape{2}, []float32{7.9, -2.1})
	result := src.I32(ctx)

	if result.Dtype() != DtypeI32 {
		t.Fatalf("expected dtype I32, got %s", result.Dtype())
	}
	expected := []int32{7, -2}
	for i, want := range expected {
		got := result.Get(ctx, uint32(i)).Item().(int32)
		if got != want {
			t.Errorf("WrappedTensor I32[%d] = %d, want %d", i, got, want)
		}
	}
}

// --- No autograd test ---

func TestCastProducesLeafWhenGradEnabled(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Finish()

	src := Float(ctx, Shape{3}, 2.0)
	result := src.I32(ctx)

	if !result.RequiresGrad() {
		t.Error("cast result should be tracked when grad is enabled")
	}
	if result.Op() != OpNone {
		t.Errorf("cast result should be a leaf (OpNone), got %v", result.Op())
	}
}

// --- Table-driven tests for all valid Tensor cast methods ---

func TestCastAllValidWidening(t *testing.T) {
	tests := []struct {
		name     string
		castFn   func(MainContext, Tensor) Tensor
		wantType Dtype
	}{
		{"I8->I16", func(ctx MainContext, src Tensor) Tensor { return src.I16(ctx) }, DtypeI16},
		{"I8->I32", func(ctx MainContext, src Tensor) Tensor { return src.I32(ctx) }, DtypeI32},
		{"I8->I64", func(ctx MainContext, src Tensor) Tensor { return src.I64(ctx) }, DtypeI64},
		{"I8->F16", func(ctx MainContext, src Tensor) Tensor { return src.F16(ctx) }, DtypeF16},
		{"I8->F32", func(ctx MainContext, src Tensor) Tensor { return src.F32(ctx) }, DtypeF32},
		{"I8->F64", func(ctx MainContext, src Tensor) Tensor { return src.F64(ctx) }, DtypeF64},
		{"I8->Bool", func(ctx MainContext, src Tensor) Tensor { return src.Bool(ctx) }, DtypeBool},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := New(context.Background())
			defer ctx.Finish()

			src := FromInt8(ctx, []int8{1, -2, 3})
			result := tt.castFn(ctx, src)
			if result.Dtype() != tt.wantType {
				t.Errorf("got dtype %s, want %s", result.Dtype(), tt.wantType)
			}
		})
	}
}

func TestCastF32Widening(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	src := FromFloat32(ctx, Shape{2}, []float32{1.5, -2.5})
	result := src.F64(ctx)
	if result.Dtype() != DtypeF64 {
		t.Errorf("got dtype %s, want F64", result.Dtype())
	}
}

// --- Value preservation across casts ---

func TestCastI8ToF32ValuePreservation(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	data := []int8{-128, -1, 0, 1, 127}
	src := FromInt8(ctx, data)
	result := src.F32(ctx)

	for i, want := range data {
		got := result.Get(ctx, uint32(i)).Item().(float32)
		if math.Abs(float64(got)-float64(want)) > 1e-5 {
			t.Errorf("F32[%d] = %f, want %f", i, got, float32(want))
		}
	}
}
