package shapes

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/build/openblas/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "cwrappers.h"
#include <string.h>
*/
import "C"

func TensorFromImagebyes(ctx Context, dims Shape, imageData []byte) Tensor {
	numChanels := dims[0]

	if len(imageData)%int(numChanels) != 0 {
		panic("Shape and image data mismatch channes do not devide data cleanly")
	}

	if numChanels == 2 {
		panic("Images should either be 1 channel or 3")
	}

	data := make([]float32, len(imageData))
	for i, value := range imageData {
		data[i] = float32(value) / 255.0
	}

	t := FromFloat32(ctx, dims, data)
	return t
}
