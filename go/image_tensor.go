package shapes

/*
#include "cwrappers.h"
#include <string.h>
*/
import "C"

func TensorFromImagebyes(ctx Context, dims Shape, imageData []byte) Tensor {
	if len(dims) == 0 {
		panic("Image tensor dims cannot be empty")
	}

	numChanels := dims[len(dims)-1]
	if len(dims) >= 3 && (dims[0] == 1 || dims[0] == 3 || dims[0] == 4) {
		numChanels = dims[0]
	}

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
