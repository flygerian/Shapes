package shapes

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/build/openblas/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "cwrappers.h"
#include <string.h>
*/
import "C"
import (
	"unsafe"
)

func TensorFromImagebyes(ctx Context, dims Shape, imageData []byte) Tensor {
	numChanels := dims[0]

	if len(imageData)%int(numChanels) != 0 {
		panic("Shape and image data mismatch channes do not devide data cleanly")
	}

	if numChanels == 2 {
		panic("Please either provide")
	}

	t := Int8(ctx, dims, 0)
	C.memcpy(t.(*tensor).cTensor.values, unsafe.Pointer(&imageData[0]), C.size_t(len(imageData)))

	return t
}
