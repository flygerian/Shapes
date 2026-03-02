package shapes

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/build/openblas/lib -lshapes_core -lshapes_memory -lopenblas -lm

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

type hasCastOps interface {
	F16(ctx Context) Tensor
	F32(ctx Context) Tensor
	F64(ctx Context) Tensor
	U8(ctx Context) Tensor
	U16(ctx Context) Tensor
	U32(ctx Context) Tensor
	U64(ctx Context) Tensor
	I8(ctx Context) Tensor
	I16(ctx Context) Tensor
	I32(ctx Context) Tensor
	I64(ctx Context) Tensor
}

// castTensor is the internal helper that calls the C Cast function.
func castTensor(ctx Context, t *tensor, target Dtype) Tensor {
	var dest *C.Tensor
	result := C.wrap_Cast((*C.Context)(ctx.UnsafePtr()), t.cTensor, C.Dtype(target), &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &tensor{cTensor: dest})
}

// --- Tensor methods ---

// F16 casts the tensor to F16 dtype. Panics if the cast would truncate or cross sign families.
func (t *tensor) F16(ctx Context) Tensor {
	return castTensor(ctx, t, DtypeF16)
}

// F32 casts the tensor to F32 dtype. Panics if the cast would truncate or cross sign families.
func (t *tensor) F32(ctx Context) Tensor {
	return castTensor(ctx, t, DtypeF32)
}

// F64 casts the tensor to F64 dtype. Panics if the cast would truncate or cross sign families.
func (t *tensor) F64(ctx Context) Tensor {
	return castTensor(ctx, t, DtypeF64)
}

// U8 casts the tensor to U8 dtype. Panics if the cast would truncate or cross sign families.
func (t *tensor) U8(ctx Context) Tensor {
	return castTensor(ctx, t, DtypeU8)
}

// U16 casts the tensor to U16 dtype. Panics if the cast would truncate or cross sign families.
func (t *tensor) U16(ctx Context) Tensor {
	return castTensor(ctx, t, DtypeU16)
}

// U32 casts the tensor to U32 dtype. Panics if the cast would truncate or cross sign families.
func (t *tensor) U32(ctx Context) Tensor {
	return castTensor(ctx, t, DtypeU32)
}

// U64 casts the tensor to U64 dtype. Panics if the cast would truncate or cross sign families.
func (t *tensor) U64(ctx Context) Tensor {
	return castTensor(ctx, t, DtypeU64)
}

// I8 casts the tensor to I8 dtype. Panics if the cast would truncate or cross sign families.
func (t *tensor) I8(ctx Context) Tensor {
	return castTensor(ctx, t, DtypeI8)
}

// I16 casts the tensor to I16 dtype. Panics if the cast would truncate or cross sign families.
func (t *tensor) I16(ctx Context) Tensor {
	return castTensor(ctx, t, DtypeI16)
}

// I32 casts the tensor to I32 dtype. Panics if the cast would truncate or cross sign families.
func (t *tensor) I32(ctx Context) Tensor {
	return castTensor(ctx, t, DtypeI32)
}

// I64 casts the tensor to I64 dtype. Panics if the cast would truncate or cross sign families.
func (t *tensor) I64(ctx Context) Tensor {
	return castTensor(ctx, t, DtypeI64)
}
