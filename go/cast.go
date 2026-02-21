package shapes

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "tensor/tensor.h"
#include "common.h"

static inline Result wrap_Cast(Context *ctx, Tensor *src, Dtype target, Tensor **out) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result r = Cast(ctx, src, dest, target);
	*out = dest;
	return r;
}
*/
import "C"

// castTensor is the internal helper that calls the C Cast function.
func castTensor(ctx *Context, t *Tensor, target Dtype) *Tensor {
	var dest *C.Tensor
	result := C.wrap_Cast((*C.Context)(ctx.UnsafePtr()), t.cTensor, C.Dtype(target), &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &Tensor{cTensor: dest})
}

// --- Tensor methods ---

// F16 casts the tensor to F16 dtype. Panics if the cast would truncate or cross sign families.
func (t *Tensor) F16(ctx *Context) *Tensor {
	return castTensor(ctx, t, DtypeF16)
}

// F32 casts the tensor to F32 dtype. Panics if the cast would truncate or cross sign families.
func (t *Tensor) F32(ctx *Context) *Tensor {
	return castTensor(ctx, t, DtypeF32)
}

// F64 casts the tensor to F64 dtype. Panics if the cast would truncate or cross sign families.
func (t *Tensor) F64(ctx *Context) *Tensor {
	return castTensor(ctx, t, DtypeF64)
}

// U8 casts the tensor to U8 dtype. Panics if the cast would truncate or cross sign families.
func (t *Tensor) U8(ctx *Context) *Tensor {
	return castTensor(ctx, t, DtypeU8)
}

// U16 casts the tensor to U16 dtype. Panics if the cast would truncate or cross sign families.
func (t *Tensor) U16(ctx *Context) *Tensor {
	return castTensor(ctx, t, DtypeU16)
}

// U32 casts the tensor to U32 dtype. Panics if the cast would truncate or cross sign families.
func (t *Tensor) U32(ctx *Context) *Tensor {
	return castTensor(ctx, t, DtypeU32)
}

// U64 casts the tensor to U64 dtype. Panics if the cast would truncate or cross sign families.
func (t *Tensor) U64(ctx *Context) *Tensor {
	return castTensor(ctx, t, DtypeU64)
}

// I8 casts the tensor to I8 dtype. Panics if the cast would truncate or cross sign families.
func (t *Tensor) I8(ctx *Context) *Tensor {
	return castTensor(ctx, t, DtypeI8)
}

// I16 casts the tensor to I16 dtype. Panics if the cast would truncate or cross sign families.
func (t *Tensor) I16(ctx *Context) *Tensor {
	return castTensor(ctx, t, DtypeI16)
}

// I32 casts the tensor to I32 dtype. Panics if the cast would truncate or cross sign families.
func (t *Tensor) I32(ctx *Context) *Tensor {
	return castTensor(ctx, t, DtypeI32)
}

// I64 casts the tensor to I64 dtype. Panics if the cast would truncate or cross sign families.
func (t *Tensor) I64(ctx *Context) *Tensor {
	return castTensor(ctx, t, DtypeI64)
}

// --- WrappedTensor methods ---

// F16 casts the wrapped tensor to F16 dtype. Panics if the cast would truncate or cross sign families.
func (wt *WrappedTensor) F16() *WrappedTensor {
	return wt.context.Wrap(wt.tensor.F16(wt.context))
}

// F32 casts the wrapped tensor to F32 dtype. Panics if the cast would truncate or cross sign families.
func (wt *WrappedTensor) F32() *WrappedTensor {
	return wt.context.Wrap(wt.tensor.F32(wt.context))
}

// F64 casts the wrapped tensor to F64 dtype. Panics if the cast would truncate or cross sign families.
func (wt *WrappedTensor) F64() *WrappedTensor {
	return wt.context.Wrap(wt.tensor.F64(wt.context))
}

// U8 casts the wrapped tensor to U8 dtype. Panics if the cast would truncate or cross sign families.
func (wt *WrappedTensor) U8() *WrappedTensor {
	return wt.context.Wrap(wt.tensor.U8(wt.context))
}

// U16 casts the wrapped tensor to U16 dtype. Panics if the cast would truncate or cross sign families.
func (wt *WrappedTensor) U16() *WrappedTensor {
	return wt.context.Wrap(wt.tensor.U16(wt.context))
}

// U32 casts the wrapped tensor to U32 dtype. Panics if the cast would truncate or cross sign families.
func (wt *WrappedTensor) U32() *WrappedTensor {
	return wt.context.Wrap(wt.tensor.U32(wt.context))
}

// U64 casts the wrapped tensor to U64 dtype. Panics if the cast would truncate or cross sign families.
func (wt *WrappedTensor) U64() *WrappedTensor {
	return wt.context.Wrap(wt.tensor.U64(wt.context))
}

// I8 casts the wrapped tensor to I8 dtype. Panics if the cast would truncate or cross sign families.
func (wt *WrappedTensor) I8() *WrappedTensor {
	return wt.context.Wrap(wt.tensor.I8(wt.context))
}

// I16 casts the wrapped tensor to I16 dtype. Panics if the cast would truncate or cross sign families.
func (wt *WrappedTensor) I16() *WrappedTensor {
	return wt.context.Wrap(wt.tensor.I16(wt.context))
}

// I32 casts the wrapped tensor to I32 dtype. Panics if the cast would truncate or cross sign families.
func (wt *WrappedTensor) I32() *WrappedTensor {
	return wt.context.Wrap(wt.tensor.I32(wt.context))
}

// I64 casts the wrapped tensor to I64 dtype. Panics if the cast would truncate or cross sign families.
func (wt *WrappedTensor) I64() *WrappedTensor {
	return wt.context.Wrap(wt.tensor.I64(wt.context))
}
