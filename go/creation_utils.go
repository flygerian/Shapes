package shapes

// flatten2D flattens a 2D slice of any type into a 1D slice.
func flatten2D[T any](data [][]T) []T {
	if len(data) == 0 {
		return nil
	}
	innerLen := len(data[0])
	flat := make([]T, 0, len(data)*innerLen)
	for _, row := range data {
		flat = append(flat, row...)
	}
	return flat
}

// flatten3D flattens a 3D slice of any type into a 1D slice.
func flatten3D[T any](data [][][]T) []T {
	if len(data) == 0 || len(data[0]) == 0 {
		return nil
	}
	dim1Len, dim2Len := len(data[0]), len(data[0][0])
	flat := make([]T, 0, len(data)*dim1Len*dim2Len)
	for _, matrix := range data {
		for _, row := range matrix {
			flat = append(flat, row...)
		}
	}
	return flat
}

// flatten4D flattens a 4D slice of any type into a 1D slice.
func flatten4D[T any](data [][][][]T) []T {
	if len(data) == 0 || len(data[0]) == 0 || len(data[0][0]) == 0 {
		return nil
	}
	dim1Len, dim2Len, dim3Len := len(data[0]), len(data[0][0]), len(data[0][0][0])
	flat := make([]T, 0, len(data)*dim1Len*dim2Len*dim3Len)
	for _, tensor3d := range data {
		for _, matrix := range tensor3d {
			for _, row := range matrix {
				flat = append(flat, row...)
			}
		}
	}
	return flat
}

// validate2D checks that a 2D slice has uniform inner dimensions.
// Returns (outerLen, innerLen, isEmpty).
// Panics if ragged array detected.
func validate2D[T any](data [][]T) (int, int, bool) {
	if len(data) == 0 {
		return 0, 0, true
	}
	expectedLen := len(data[0])
	for i, row := range data {
		if len(row) != expectedLen {
			panic("shapes: ragged array detected in row " + string(rune('0'+i)) +
				", expected length " + string(rune('0'+expectedLen)) + " but got " + string(rune('0'+len(row))))
		}
	}
	if expectedLen == 0 {
		return len(data), 0, true
	}
	return len(data), expectedLen, false
}

// validate3D checks that a 3D slice has uniform dimensions.
// Returns (dim0, dim1, dim2, isEmpty).
// Panics if ragged array detected.
func validate3D[T any](data [][][]T) (int, int, int, bool) {
	if len(data) == 0 {
		return 0, 0, 0, true
	}
	dim1Len := len(data[0])
	for i, matrix := range data {
		if len(matrix) != dim1Len {
			panic("shapes: ragged array detected at dimension 1, index " + string(rune('0'+i)))
		}
	}
	if dim1Len == 0 {
		return len(data), 0, 0, true
	}
	dim2Len := len(data[0][0])
	for i, matrix := range data {
		for j, row := range matrix {
			if len(row) != dim2Len {
				panic("shapes: ragged array detected at dimension 2, index [" + string(rune('0'+i)) + "][" + string(rune('0'+j)) + "]")
			}
		}
	}
	if dim2Len == 0 {
		return len(data), dim1Len, 0, true
	}
	return len(data), dim1Len, dim2Len, false
}

// validate4D checks that a 4D slice has uniform dimensions.
// Returns (dim0, dim1, dim2, dim3, isEmpty).
// Panics if ragged array detected.
func validate4D[T any](data [][][][]T) (int, int, int, int, bool) {
	if len(data) == 0 {
		return 0, 0, 0, 0, true
	}
	dim1Len := len(data[0])
	for i, tensor3d := range data {
		if len(tensor3d) != dim1Len {
			panic("shapes: ragged array detected at dimension 1, index " + string(rune('0'+i)))
		}
	}
	if dim1Len == 0 {
		return len(data), 0, 0, 0, true
	}
	dim2Len := len(data[0][0])
	for i, tensor3d := range data {
		for j, matrix := range tensor3d {
			if len(matrix) != dim2Len {
				panic("shapes: ragged array detected at dimension 2, index [" + string(rune('0'+i)) + "][" + string(rune('0'+j)) + "]")
			}
		}
	}
	if dim2Len == 0 {
		return len(data), dim1Len, 0, 0, true
	}
	dim3Len := len(data[0][0][0])
	for _, tensor3d := range data {
		for _, matrix := range tensor3d {
			for _, row := range matrix {
				if len(row) != dim3Len {
					panic("shapes: ragged array detected at dimension 3")
				}
			}
		}
	}
	if dim3Len == 0 {
		return len(data), dim1Len, dim2Len, 0, true
	}
	return len(data), dim1Len, dim2Len, dim3Len, false
}
