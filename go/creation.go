package shapes

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "tensor/tensor.h"
#include "common.h"
#include <string.h>

static inline Tensor *wrap_T_Zeros(Context *ctx, Dim *shape) {
	Tensor *t = T_Zeros(ctx, *shape);
	freeAlloc(ctx->memory, shape->dims);
	freeAlloc(ctx->memory, shape);
	return t;
}
static inline Tensor *wrap_T_Int(Context *ctx, Dim *shape, i8 value) {
	Tensor *t = T_Int(ctx, *shape, value);
	freeAlloc(ctx->memory, shape->dims);
	freeAlloc(ctx->memory, shape);
	return t;
}
static inline Tensor *wrap_T_Float(Context *ctx, Dim *shape, f32 value) {
	Tensor *t = T_Float(ctx, *shape, value);
	freeAlloc(ctx->memory, shape->dims);
	freeAlloc(ctx->memory, shape);
	return t;
}
static inline Result wrap_Clone(Context *ctx, Tensor *src, Tensor **out) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result r = Clone(ctx, src, dest);
	*out = dest;
	return r;
}
static inline Tensor *wrap_T_OneHot(Context *ctx, Tensor *indices, dim_t numClasses) {
	return T_OneHot(ctx, indices, numClasses);
}
static inline Tensor *wrap_T_Arange(Context *ctx, f32 start, f32 end, f32 step) {
	return T_Arange(ctx, start, end, step);
}
*/
import "C"
import (
	"math/rand"
	"unsafe"
)

// Zeros creates a tensor filled with zeros.
func Zeros(ctx *Context, shape Shape) *Tensor {
	if len(shape) == 0 {
		return nil
	}
	cTensor := C.wrap_T_Zeros((*C.Context)(ctx.UnsafePtr()), dim(ctx, shape))
	t := track(ctx, &Tensor{cTensor: cTensor})
	if ctx.GradEnabled {
		leafNode(ctx, t)
	}
	return t
}

// Int creates a tensor filled with the given int8 value.
func Int(ctx *Context, shape Shape, value int8) *Tensor {
	if len(shape) == 0 {
		return nil
	}
	cTensor := C.wrap_T_Int((*C.Context)(ctx.UnsafePtr()), dim(ctx, shape), C.i8(value))
	t := track(ctx, &Tensor{cTensor: cTensor})
	if ctx.GradEnabled {
		leafNode(ctx, t)
	}
	return t
}

// Float creates a tensor filled with the given float32 value.
func Float(ctx *Context, shape Shape, value float32) *Tensor {
	if len(shape) == 0 {
		return nil
	}
	cTensor := C.wrap_T_Float((*C.Context)(ctx.UnsafePtr()), dim(ctx, shape), C.f32(value))
	t := track(ctx, &Tensor{cTensor: cTensor})
	if ctx.GradEnabled {
		leafNode(ctx, t)
	}
	return t
}

// FromFloat32 creates a tensor from a Go []float32 slice with the given shape.
// Panics if the number of elements in data does not match the shape.
func FromFloat32(ctx *Context, shape Shape, data []float32) *Tensor {
	if len(shape) == 0 {
		return nil
	}

	t := Zeros(ctx, shape)

	expected := int(t.cTensor.size)
	if len(data) != expected {
		panic("shapes: data length does not match shape")
	}

	C.memcpy(t.cTensor.values, unsafe.Pointer(&data[0]), C.size_t(expected)*C.sizeof_float)

	if ctx.GradEnabled {
		leafNode(ctx, t)
	}

	return t
}

// fromInt8_1D converts a 1D int8 slice to shape and flat data.
// Returns true if data is empty (nil should be returned).
func fromInt8_1D(data []int8) (Shape, []int8, bool) {
	if len(data) == 0 {
		return nil, nil, true
	}
	return Shape{uint32(len(data))}, data, false
}

// fromInt8_2D converts a 2D int8 slice to shape and flat data.
// Panics if inner slices have inconsistent lengths.
// Returns true if data is empty (nil should be returned).
func fromInt8_2D(data [][]int8) (Shape, []int8, bool) {
	d0, d1, isEmpty := validate2D(data)
	if isEmpty {
		return nil, nil, true
	}
	shape := Shape{uint32(d0), uint32(d1)}
	flatData := flatten2D(data)
	return shape, flatData, false
}

// fromInt8_3D converts a 3D int8 slice to shape and flat data.
// Panics if nested slices have inconsistent dimensions.
// Returns true if data is empty (nil should be returned).
func fromInt8_3D(data [][][]int8) (Shape, []int8, bool) {
	d0, d1, d2, isEmpty := validate3D(data)
	if isEmpty {
		return nil, nil, true
	}
	shape := Shape{uint32(d0), uint32(d1), uint32(d2)}
	flatData := flatten3D(data)
	return shape, flatData, false
}

// fromInt8_4D converts a 4D int8 slice to shape and flat data.
// Panics if nested slices have inconsistent dimensions.
// Returns true if data is empty (nil should be returned).
func fromInt8_4D(data [][][][]int8) (Shape, []int8, bool) {
	d0, d1, d2, d3, isEmpty := validate4D(data)
	if isEmpty {
		return nil, nil, true
	}
	shape := Shape{uint32(d0), uint32(d1), uint32(d2), uint32(d3)}
	flatData := flatten4D(data)
	return shape, flatData, false
}

// FromInt8 creates a tensor from nested int8 slices, inferring the shape from the data structure.
// Supports up to 4D tensors: []int8 (1D), [][]int8 (2D), [][][]int8 (3D), [][][][]int8 (4D).
// Panics if nested slices have inconsistent lengths (ragged arrays).
// Returns nil for empty data.
func FromInt8(ctx *Context, data interface{}) *Tensor {
	var shape Shape
	var flatData []int8
	var isEmpty bool

	switch d := data.(type) {
	case []int8:
		shape, flatData, isEmpty = fromInt8_1D(d)
	case [][]int8:
		shape, flatData, isEmpty = fromInt8_2D(d)
	case [][][]int8:
		shape, flatData, isEmpty = fromInt8_3D(d)
	case [][][][]int8:
		shape, flatData, isEmpty = fromInt8_4D(d)
	default:
		panic("shapes: unsupported type for FromInt8, expected []int8, [][]int8, [][][]int8, or [][][][]int8")
	}

	if isEmpty {
		return nil
	}

	t := Int(ctx, shape, 0)

	C.memcpy(t.cTensor.values, unsafe.Pointer(&flatData[0]), C.size_t(len(flatData)))

	if ctx.GradEnabled {
		leafNode(ctx, t)
	}

	return t
}

// FloatRandom creates a tensor with random float32 values.
// If no range is provided, values are uniformly distributed in [-1, 1].
// If min and max are provided, values are uniformly distributed in [min, max].
// Panics if only min is provided without max.
func FloatRandom(ctx *Context, shape Shape, rng ...float32) *Tensor {
	if len(shape) == 0 {
		return nil
	}

	// Parse range argument
	var min, max float32
	switch len(rng) {
	case 0:
		// Default range: [-1, 1]
		min, max = -1, 1
	case 2:
		// Custom range: [min, max]
		min, max = rng[0], rng[1]
		if min > max {
			panic("shapes: FloatRandom requires min <= max")
		}
	default:
		panic("shapes: FloatRandom requires 0 or 2 range arguments (min, max)")
	}

	t := Zeros(ctx, shape)
	n := int(t.cTensor.size)
	data := make([]float32, n)
	scale := max - min
	for i := range n {
		data[i] = rand.Float32()*scale + min
	}
	C.memcpy(t.cTensor.values, unsafe.Pointer(&data[0]), C.size_t(n)*C.sizeof_float)

	if ctx.GradEnabled {
		leafNode(ctx, t)
	}

	return t
}

// IntRandom creates a tensor with random int8 values.
// If no range is provided, values are uniformly distributed in [-128, 127] (full int8 range).
// If min and max are provided, values are uniformly distributed in [min, max] (inclusive).
// Panics if only min is provided without max.
func IntRandom(ctx *Context, shape Shape, rng ...int8) *Tensor {
	if len(shape) == 0 {
		return nil
	}

	// Parse range argument
	var min, max int8
	switch len(rng) {
	case 0:
		// Default range: full int8 range [-128, 127]
		min, max = -128, 127
	case 2:
		// Custom range: [min, max]
		min, max = rng[0], rng[1]
		if min > max {
			panic("shapes: IntRandom requires min <= max")
		}
	default:
		panic("shapes: IntRandom requires 0 or 2 range arguments (min, max)")
	}

	t := Int(ctx, shape, 0)
	n := int(t.cTensor.size)
	data := make([]int8, n)
	rangeSize := int(max) - int(min) + 1 // +1 because max is inclusive
	for i := range n {
		data[i] = int8(rand.Intn(rangeSize) + int(min))
	}
	C.memcpy(t.cTensor.values, unsafe.Pointer(&data[0]), C.size_t(n))

	if ctx.GradEnabled {
		leafNode(ctx, t)
	}

	return t
}

// Free releases the tensor's C memory back to the arena.
// The tensor must not be a view. After Free, the tensor must not be used.
func (t *Tensor) Free(ctx *Context) {
	result := C.FreeTensor((*C.Context)(ctx.UnsafePtr()), t.cTensor)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
}

// FreeView releases a view tensor's metadata (dims, multipliers, boundary)
// back to the arena without freeing the shared values.
func (t *Tensor) FreeView(ctx *Context) {
	result := C.FreeViewTensor((*C.Context)(ctx.UnsafePtr()), t.cTensor)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
}

// Free releases the WrappedTensor's C memory back to the arena.
// After Free, the WrappedTensor must not be used.
func (wt *WrappedTensor) Free() {
	wt.tensor.Free(wt.context)
}

// FreeView releases a view WrappedTensor's metadata back to the arena
// without freeing the shared values.
func (wt *WrappedTensor) FreeView() {
	wt.tensor.FreeView(wt.context)
}

// --- Context creation methods ---

// Zeros creates a tensor filled with zeros on the context, returning a WrappedTensor.
// Returns nil for empty shape.
func (c *Context) Zeros(shape Shape) *WrappedTensor {
	t := Zeros(c, shape)
	if t == nil {
		return nil
	}
	return c.Wrap(t)
}

// Int creates a tensor filled with the given int8 value on the context, returning a WrappedTensor.
// Returns nil for empty shape.
func (c *Context) Int(shape Shape, value int8) *WrappedTensor {
	t := Int(c, shape, value)
	if t == nil {
		return nil
	}
	return c.Wrap(t)
}

// Float creates a tensor filled with the given float32 value on the context, returning a WrappedTensor.
// Returns nil for empty shape.
func (c *Context) Float(shape Shape, value float32) *WrappedTensor {
	t := Float(c, shape, value)
	if t == nil {
		return nil
	}
	return c.Wrap(t)
}

// FromFloat32 creates a tensor from a Go []float32 slice on the context, returning a WrappedTensor.
// Panics if the number of elements in data does not match the shape.
// Returns nil for empty shape.
func (c *Context) FromFloat32(shape Shape, data []float32) *WrappedTensor {
	t := FromFloat32(c, shape, data)
	if t == nil {
		return nil
	}
	return c.Wrap(t)
}

// FromInt8 creates a tensor from nested int8 slices on the context, returning a WrappedTensor.
// Supports up to 4D tensors: []int8 (1D), [][]int8 (2D), [][][]int8 (3D), [][][][]int8 (4D).
// Panics if nested slices have inconsistent lengths (ragged arrays).
// Returns nil for empty data.
func (c *Context) FromInt8(data interface{}) *WrappedTensor {
	t := FromInt8(c, data)
	if t == nil {
		return nil
	}
	return c.Wrap(t)
}

// FloatRandom creates a tensor with random float32 values on the context, returning a WrappedTensor.
// If no range is provided, values are uniformly distributed in [-1, 1].
// If min and max are provided, values are uniformly distributed in [min, max].
// Panics if only min is provided without max.
// Returns nil for empty shape.
func (c *Context) FloatRandom(shape Shape, rng ...float32) *WrappedTensor {
	t := FloatRandom(c, shape, rng...)
	if t == nil {
		return nil
	}
	return c.Wrap(t)
}

// IntRandom creates a tensor with random int8 values on the context, returning a WrappedTensor.
// If no range is provided, values are uniformly distributed in [-128, 127] (full int8 range).
// If min and max are provided, values are uniformly distributed in [min, max] (inclusive).
// Panics if only min is provided without max.
// Returns nil for empty shape.
func (c *Context) IntRandom(shape Shape, rng ...int8) *WrappedTensor {
	t := IntRandom(c, shape, rng...)
	if t == nil {
		return nil
	}
	return c.Wrap(t)
}

// Clone creates a deep copy of the given tensor on the context, returning a WrappedTensor.
func (c *Context) Clone(src *WrappedTensor) *WrappedTensor {
	return c.Wrap(Clone(c, src.tensor))
}

// OneHot creates a one-hot encoded tensor from indices on the context, returning a WrappedTensor.
// Returns nil for nil indices or zero numClasses.
func (c *Context) OneHot(indices *WrappedTensor, numClasses uint32) *WrappedTensor {
	t := OneHot(c, indices.tensor, numClasses)
	if t == nil {
		return nil
	}
	return c.Wrap(t)
}

// Clone creates a deep copy of the tensor.
func Clone(ctx *Context, src *Tensor) *Tensor {
	var dest *C.Tensor
	result := C.wrap_Clone((*C.Context)(ctx.UnsafePtr()), src.cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &Tensor{cTensor: dest})
}

// OneHot creates a one-hot encoded tensor from indices.
// The input tensor contains class indices, and the output will have an additional
// dimension of size numClasses where each index is represented as a one-hot vector.
// For example, indices [[0, 2], [1, 0]] with numClasses=3 becomes:
// [[[1,0,0], [0,0,1]], [[0,1,0], [1,0,0]]]
func OneHot(ctx *Context, indices *Tensor, numClasses uint32) *Tensor {
	if indices == nil || numClasses == 0 {
		return nil
	}
	cTensor := C.wrap_T_OneHot((*C.Context)(ctx.UnsafePtr()), indices.cTensor, C.dim_t(numClasses))
	if cTensor == nil {
		return nil
	}
	t := track(ctx, &Tensor{cTensor: cTensor})
	if ctx.GradEnabled {
		ctx.newNode(t, OpOneHot, oneHotBackward, nil, indices)
	}
	return t
}

// Arange creates a 1D tensor with values from start to end (exclusive) with the given step.
// Similar to PyTorch's torch.arange.
// Supports variadic arguments:
//   - Arange(ctx, end): range from 0 to end with step 1
//   - Arange(ctx, start, end): range from start to end with step 1
//   - Arange(ctx, start, end, step): range from start to end with given step
//
// Returns nil if the range is empty or invalid.
func Arange(ctx *Context, args ...float32) *Tensor {
	var start, end, step float32

	switch len(args) {
	case 1:
		// arange(end): start=0, end=args[0], step=1
		start = 0
		end = args[0]
		step = 1
	case 2:
		// arange(start, end): start=args[0], end=args[1], step=1
		start = args[0]
		end = args[1]
		step = 1
	case 3:
		// arange(start, end, step)
		start = args[0]
		end = args[1]
		step = args[2]
	default:
		panic("shapes: Arange requires 1, 2, or 3 arguments")
	}

	cTensor := C.wrap_T_Arange((*C.Context)(ctx.UnsafePtr()), C.f32(start), C.f32(end), C.f32(step))
	if cTensor == nil {
		return nil
	}
	t := track(ctx, &Tensor{cTensor: cTensor})
	if ctx.GradEnabled {
		leafNode(ctx, t)
	}
	return t
}

// Arange creates a 1D tensor with values from start to end (exclusive) with the given step,
// returning a WrappedTensor. Similar to PyTorch's torch.arange.
// Supports variadic arguments:
//   - ctx.Arange(end): range from 0 to end with step 1
//   - ctx.Arange(start, end): range from start to end with step 1
//   - ctx.Arange(start, end, step): range from start to end with given step
//
// Returns nil if the range is empty or invalid.
func (c *Context) Arange(args ...float32) *WrappedTensor {
	t := Arange(c, args...)
	if t == nil {
		return nil
	}
	return c.Wrap(t)
}
