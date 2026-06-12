package shapes

import "core:c"
import "core:mem"
import "olib"

foreign import shapes {"../../build/libshapes.a", "../../build/libolib.a", "../../build/openblas-install/lib/libopenblas.a"}

CudaMemory :: struct {
	blocks:               ^olib.Array,
	allocationPointer:    uint,
	allocationCheckpoint: i64,
}

DeviceType :: enum c.int {
	CPU,
	CUDA,
}

Device :: struct {
	type: DeviceType,
	id:   cstring,
}

Context :: struct {
	memory:             ^olib.Memory,
	cudaMetadataMemory: ^olib.Memory,
	cudaMemory:         CudaMemory,
	device:             ^Device,
	isTraining:         bool,
	handle:             rawptr,
	parent:             ^Context,
}

Raw_Dim :: struct {
	dims:        [^]uint,
	multipliers: [^]uint,
	numOfDims:   u8,
}

Dim :: struct {
	dims:        []uint,
	multipliers: []uint,
}

Range :: struct {
	start: uint,
	end:   uint,
}

Dtype :: enum c.int {
	F16,
	F32,
	F64,
	U8,
	U16,
	U32,
	U64,
	I8,
	I16,
	I32,
	I64,
	BOOL,
}

ValueAs :: struct #raw_union {
	boolean: bool,
	u8_val:  u8,
	u16_val: u16,
	u32_val: u32,
	u64_val: u64,
	i8_val:  i8,
	i16_val: i16,
	i32_val: i32,
	i64_val: i64,
	f16_val: f32,
	f32_val: f32,
	f64_val: f64,
}

Raw_Value :: struct {
	dtype: Dtype,
	as:    ValueAs,
}

Optype :: enum c.int {
	OP_NONE,
	OP_ADD,
	OP_SUBTRACT,
	OP_MULTIPLY,
	OP_GREATER,
	OP_GREATER_OR_EQUAL,
	OP_LESS,
	OP_LESS_OR_EQUAL,
	OP_EQUAL,
	OP_DENSE,
	OP_EMBEDDING,
	OP_RESHAPE,
	OP_MSE,
	OP_CROSS_ENTHROPY,
	OP_SGD,
	OP_ADAM,
	OP_BATCH_NORM,
	OP_TANH,
	OP_SQRT,
	OP_RELU,
	OP_MAXPOOL2D,
	OP_ADAPTIVE_AVG_POOL2D,
	OP_CONV2D,
	OP_SEQUENTIAL,
	OP_FLATTEN,
}

Raw_BatchNormForwardResult :: struct {
	out:      Raw_Tensor,
	mean:     Raw_Tensor,
	variance: Raw_Tensor,
}

Raw_BatchNormBackwardResult :: struct {
	dx2d:   Raw_Tensor,
	dGamma: Raw_Tensor,
	dBeta:  Raw_Tensor,
}

Raw_TensorPair :: struct {
	a: Raw_Tensor,
	b: Raw_Tensor,
}

Raw_AdamData :: struct {
	param: rawptr,
	grad:  rawptr,
	m:     rawptr,
	v:     rawptr,
	size:  uint,
	dtype: Dtype,
}

TensorValue :: union #no_nil {
	f16,
	f32,
	f64,
	i8,
	i16,
	i32,
	i64,
}

Tensor :: struct {
	ctx:             ^Context,
	label:           cstring,
	metadataMemory:  ^olib.Memory,
	values:          []TensorValue,
	dtype:           Dtype,
	boundary:        ^Range,
	size:            uint,
	shape:           Dim,
	isView:          bool,
	isContigous:     bool,
	isContigousCopy: bool,
	grad:            ^Tensor,
	nodeId:          u64,
}

Raw_Tensor :: struct {
	ctx:             ^Context,
	label:           cstring,
	metadataMemory:  ^olib.Memory,
	values:          rawptr,
	boundary:        ^Range,
	size:            uint,
	shape:           Raw_Dim,
	dtype:           Dtype,
	isView:          bool,
	isContigous:     bool,
	isContigousCopy: bool,
	grad:            rawptr,
	inputs:          ^olib.Array,
	opType:          Optype,
	opMetadata:      rawptr,
	nodeId:          u64,
}

Result :: enum c.int {
	OK,
	ERR_NO_OP,
	ALLOCATION_FAILED,
	CUDA_OP_FAILED,
	NULL_CONTEXT,
	NO_DEVICE_ON_CONTEXT,
	ERR_DEVICE_MISMATCH,
	ALLOCATING_ZERO,
	ERR_DTYPE_MISMATCH,
	ERR_DIM_MISMATCH,
	ERR_OUT_OF_BOUNDS,
	ERR_SUM_DIM_OUT_OF_BOUNDS,
	ERR_NULL_PTR,
	ERR_INVALID_RANGE,
	ERR_NULL_TENSOR_PROVIDED,
	ERR_NULL_SHAPE_PROVIDED,
	ERR_RESHAPE_DIM_MISMATCH,
	ERR_INVALID_TRANSPOSE,
	ERR_MAX_N_DIMS_EXCEEDED,
	ERR_CANNOT_FREE_VIEW_TENSOR,
	ERR_MATMUL_MIN_2D,
	ERR_MATMUL_INNER_DIM_MISMATCH,
	ERR_TANH_VALUE_NOT_FLOAT,
	ERR_RELU_VALUE_NOT_FLOAT,
	ERR_POW_VALUE_NOT_FLOAT,
	ERR_EXP_VALUE_NOT_FLOAT,
	ERR_NEGATE_UNSUPPORTED_DTYPE,
	ERR_MEAN_VALUE_NOT_FLOAT,
	ERR_LOG_VALUE_NOT_FLOAT,
	ERR_ABS_VALUE_NOT_SIGNED,
	ERR_SQRT_VALUE_NOT_FLOAT,
	ERR_NOT_A_BINOP,
	ERR_ZERO_DIM_TENSOR_ADVANCED_INDEXING,
	ERR_ONLY_INT_TYPE_ALLOWED,
	ERR_TRUNCATING_CAST,
	ERR_SIGN_MISMATCH_CAST,
	ERR_OUT_OF_MEMORY,
	ERR_COPY_REQUIRES_INITIALIZED_TENSORS,
	ERR_COPY_REQUIRES_TENSORS_OF_THE_SAME_SIZE,
	ERR_COPY_DESTINATION_VIEW,
	ERR_COPY_SAME_DTYPE,
	ERR_STD_NOT_FLOAT_TYPE,
	ERR_STD_REQUIRES_AT_LEAST_TWO_VALUES,
	ERR_LEARNING_RATE_CANNOT_BE_ZERO_OR_NEGATIVE,
	ERR_SGD_PARAMS_HAVE_TO_BE_CONTIGOUS,
	ERR_SGD_PARAMS_HAVE_TO_BE_FLOAT,
	ERR_SGD_PARAMS_NUMBER_MISMATCH,
	ERR_SGD_PARAMS_GRAD_DTYPE_MISMATCH,
	ERR_CONV2D_INVALID_NUM_TENSOR_DIM,
	ERR_CONV2D_IN_CHANNELS_ZERO,
	ERR_CONV2D_OUT_CHANNELS_ZERO,
	ERR_CONV2D_KERNEL_NOT_2D,
	ERR_CONV2D_KERNEL_NOT_FLOAT,
	ERR_CONV2D_KERNEL_STRIDE_ZERO,
	ERR_CONV2D_KERNEL_NOT_CONTIGOUS,
	ERR_ADAM_NULL_TRIPLETS,
	ERR_ADAM_NULL_M,
	ERR_ADAM_NULL_V,
	ERR_ADAM_NULL_PARAM,
	ERR_ADAM_NULL_GRAD,
	ERR_ADAM_ONLY_FLOAT_TENSORS,
	ERR_ADAM_PARAMS_SIZE_MISMATCH,
	ERR_CONCAT_TENSOR_IS_NULL,
	ERR_CONCAT_TARGET_DIM_IS_OUT_OF_BOUNDS,
	ERR_CONCAT_TENSOR_DOES_NOT_FIT_IN_TARGET_DIM,
	ERR_CONCAT_TENSOR_NOT_CONTIGOUS,
	ERR_CONCAT_TENSOR_NOT_SAME_DTYPE,
	ERR_CONCAT_TENSORS_UNEQUAL_DIMS,
	ERR_CONCAT_SOURCE_TENSOR_CANNOT_HAVE_ZERO_DIMS,
	ERR_COPY_CTX_DEVICE_IS_NULL,
	ERR_DIFFERENT_CTX_TENSORS_PASSED,
	NON_CONTIGOUS_MOVE_TENSOR,
	NOT_A_DENSE_LAYER,
	OPTIMIZER_OP_NOT_FOUND,
	LAYER_OP_NOT_FOUND,
	BACKWARD_TENSOR_OP_NOT_FOUND,
	BATCH_NORM_ZERO_DIM_NOT_ALLOWED,
	FILE_OPEN_FAILED,
	TENSORS_CANNOT_BE_BROADCASTED,
	ZERO_LAYERS_PASSED,
	NON_LAYER_OP_PASSED,
	OP_NOT_SEQUENTIAL,
	ARRAY_ELEM_SIZE_MISMATCH,
	ERR_EXPAND_FIXED_ARRAY,
	ERR_EOF,
	ERR_STACKING_LESS_THAN_TWO_TENSORS,
	ERR_CUDA_BLOCK_MISMATCH,
	ERR_CUDA_BLOCK_NO_ALLOCATION_CHECKPOINT,
}

ArrayTensor :: ^olib.Array

@(link_prefix = "shapes_")
foreign shapes {
	// Context
	InitializeHostContext :: proc(arenaSize: uint, minBlockSize: uint) -> Context ---
	InitializeCudaContext :: proc(hostArenaSize: uint) -> Context ---
	DestroyContext :: proc(ctx: ^Context) ---
	GetScratchContext :: proc(ctx: ^Context, bufferSize: uint) -> Context ---
	FreeContext :: proc(ctx: ^Context) ---

	// device
	Flush :: proc(ctx: ^Context) -> Result ---
	CopyBetweenDevices :: proc(srcType: DeviceType, destType: DeviceType, srcPtr: rawptr, destPtr: rawptr, size: uint) -> Result ---
	MoveToCuda :: proc(destCtx: ^Context, tensors: ArrayTensor) ---
	MoveToHost :: proc(destCtx: ^Context, tensors: ArrayTensor) ---
	MoveTensorToHost :: proc(destCtx: ^Context, t: ^Raw_Tensor) ---

	// Debug
	PrintItem :: proc(t: ^Raw_Tensor) ---
	@(link_name = "shapes_PrintTensor")
	_PrintTensor :: proc(tensor: ^Raw_Tensor) ---

	// Array helpers
	@(link_name = "shapes_Make_DynamicTensorArray")
	_Make_DynamicTensorArray :: proc(memory: ^olib.Memory) -> ArrayTensor ---
	@(link_name = "shapes_MakeTensorArray")
	_MakeTensorArray :: proc(memory: ^olib.Memory, capacity: uint) -> ^olib.Array ---
	@(link_name = "shapes_ArrayAppendTensor")
	_ArrayAppendTensor :: proc(array: ^olib.Array, tensor: ^Raw_Tensor) ---
	@(link_name = "shapes_ArrayAppendTensorArray")
	_ArrayAppendTensorArray :: proc(array: ^olib.Array, tensorArray: ^olib.Array) ---

	// Tensor creation
	@(link_name = "shapes_MakeZerosTensor")
	_MakeZerosTensor :: proc(ctx: ^Context, shape: Raw_Dim) -> Raw_Tensor ---
	@(link_name = "shapes_MakeZerosTensorWithDtype")
	_MakeZerosTensorWithType :: proc(ctx: ^Context, shape: Raw_Dim, dtype: Dtype) -> Raw_Tensor ---
	@(link_name = "shapes_MakeIntTensor")
	_MakeIntTensor :: proc(ctx: ^Context, shape: Raw_Dim, initialValue: i8) -> Raw_Tensor ---
	@(link_name = "shapes_MakeUIntTensor")
	_MakeUIntTensor :: proc(ctx: ^Context, shape: Raw_Dim, initialValue: u8) -> Raw_Tensor ---
	@(link_name = "shapes_MakeFloatTensor")
	_MakeFloatTensor :: proc(ctx: ^Context, shape: Raw_Dim, initialValue: f32) -> Raw_Tensor ---
	@(link_name = "shapes_MakeFloat64Tensor")
	_MakeFloat64Tensor :: proc(ctx: ^Context, shape: Raw_Dim, initialValue: f64) -> Raw_Tensor ---
	@(link_name = "shapes_MakeFromContigousArray")
	_MakeFromContigousArray :: proc(ctx: ^Context, shape: Raw_Dim, values: rawptr, dtype: Dtype) -> Raw_Tensor ---
	@(link_name = "shapes_MakeRandomTensor")
	_MakeRandomTensor :: proc(ctx: ^Context, shape: Raw_Dim, minValue: f32, maxValue: f32, dtype: Dtype) -> Raw_Tensor ---
	@(link_name = "shapes_MakeOneHotTensor")
	_MakeOneHotTensor :: proc(ctx: ^Context, indices: ^Raw_Tensor, numClasses: uint) -> Raw_Tensor ---
	@(link_name = "shapes_MakeArangeTensor")
	_MakeArangeTensor :: proc(ctx: ^Context, start: f32, end: f32, step: f32) -> Raw_Tensor ---
	@(link_name = "shapes_SetValues")
	_SetValues :: proc(t: ^Raw_Tensor, value: Raw_Value) ---

	// Binary ops
	@(link_name = "shapes_Add")
	_Add :: proc(ctx: ^Context, a: ^Raw_Tensor, b: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_Subtract")
	_Subtract :: proc(ctx: ^Context, a: ^Raw_Tensor, b: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_Divide")
	_Divide :: proc(ctx: ^Context, a: ^Raw_Tensor, b: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_Multiply")
	_Multiply :: proc(ctx: ^Context, a: ^Raw_Tensor, b: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_GreaterThan")
	_GreaterThan :: proc(ctx: ^Context, a: ^Raw_Tensor, b: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_GreaterThanOrEqual")
	_GreaterThanOrEqual :: proc(ctx: ^Context, a: ^Raw_Tensor, b: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_Equal")
	_Equal :: proc(ctx: ^Context, a: ^Raw_Tensor, b: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_LessThan")
	_LessThan :: proc(ctx: ^Context, a: ^Raw_Tensor, b: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_LessThanOrEqual")
	_LessThanOrEqual :: proc(ctx: ^Context, a: ^Raw_Tensor, b: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_MatMul")
	_MatMul :: proc(ctx: ^Context, a: ^Raw_Tensor, b: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_Dot")
	_Dot :: proc(ctx: ^Context, a: ^Raw_Tensor, b: ^Raw_Tensor) -> Raw_Tensor ---

	// Inplace ops
	@(link_name = "shapes_AddInPlace")
	_AddInPlace :: proc(ctx: ^Context, a: ^Raw_Tensor, b: ^Raw_Tensor) ---
	@(link_name = "shapes_SubtractInPlace")
	_SubtractInPlace :: proc(ctx: ^Context, a: ^Raw_Tensor, b: ^Raw_Tensor) ---
	@(link_name = "shapes_MultiplyInPlace")
	_MultiplyInPlace :: proc(ctx: ^Context, a: ^Raw_Tensor, b: ^Raw_Tensor) ---

	// Indexing
	@(link_name = "shapes_GetAt")
	_GetAt :: proc(t: ^Raw_Tensor, dim: Raw_Dim) -> ^Raw_Value ---
	@(link_name = "shapes_IndexWithTensor")
	_IndexWithTensor :: proc(ctx: ^Context, source: ^Raw_Tensor, indices: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_IndexWithTensor2d")
	_IndexWithTensor2d :: proc(ctx: ^Context, source: ^Raw_Tensor, rowIndices: ^Raw_Tensor, colIndices: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_IndexAccumulate1d")
	_IndexAccumulate1d :: proc(ctx: ^Context, dest: ^Raw_Tensor, indices: ^Raw_Tensor, srcGrad: ^Raw_Tensor) ---
	@(link_name = "shapes_IndexAccumulate2d")
	_IndexAccumulate2d :: proc(ctx: ^Context, dest: ^Raw_Tensor, rowIndices: ^Raw_Tensor, colIndices: ^Raw_Tensor, srcGrad: ^Raw_Tensor) ---
	@(link_name = "shapes_ReluBackwardAccumulate")
	_ReluBackwardAccumulate :: proc(ctx: ^Context, output: ^Raw_Tensor, gradOut: ^Raw_Tensor, dest: ^Raw_Tensor) ---

	// Shape manipulation
	@(link_name = "shapes_Reshape")
	_Reshape :: proc(ctx: ^Context, source: ^Raw_Tensor, dim: Raw_Dim) -> Raw_Tensor ---
	@(link_name = "shapes_ReshapeBackward")
	_ReshapeBackward :: proc(ctx: ^Context, node: ^Raw_Tensor) ---
	@(link_name = "shapes_Transpose")
	_Transpose :: proc(ctx: ^Context, source: ^Raw_Tensor, #c_vararg args: ..any) -> Raw_Tensor ---
	@(link_name = "shapes_Permute")
	_Permute :: proc(ctx: ^Context, source: ^Raw_Tensor, order: Raw_Dim) -> Raw_Tensor ---
	@(link_name = "shapes_Squeeze")
	_Squeeze :: proc(ctx: ^Context, t: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_SqueezeDim")
	_SqueezeDim :: proc(ctx: ^Context, t: ^Raw_Tensor, dim: uint) -> Raw_Tensor ---
	@(link_name = "shapes_UnSqueeze")
	_UnSqueeze :: proc(ctx: ^Context, t: ^Raw_Tensor, dim: uint) -> Raw_Tensor ---

	// Copying
	@(link_name = "shapes_Clone")
	_Clone :: proc(ctx: ^Context, tensor: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_Copy")
	_Copy :: proc(ctx: ^Context, src: ^Raw_Tensor, dest: ^Raw_Tensor) ---
	@(link_name = "shapes_Concat")
	_Concat :: proc(ctx: ^Context, target: ^Raw_Tensor, targetDim: uint, tensors: ArrayTensor) -> Raw_Tensor ---
	@(link_name = "shapes_Stack")
	_Stack :: proc(ctx: ^Context, tensors: ArrayTensor) -> Raw_Tensor ---

	// Unary
	@(link_name = "Cast")
	_Cast :: proc(ctx: ^Context, source: ^Raw_Tensor, targetDtype: Dtype) -> Raw_Tensor ---
	@(link_name = "shapes_Pow")
	_Pow :: proc(ctx: ^Context, t: ^Raw_Tensor, power: f32) -> Raw_Tensor ---
	@(link_name = "shapes_Exp")
	_Exp :: proc(ctx: ^Context, t: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_Tanh")
	_Tanh :: proc(ctx: ^Context, t: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_Relu")
	_Relu :: proc(ctx: ^Context, t: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_ReluBackward")
	_ReluBackward :: proc(ctx: ^Context, output: ^Raw_Tensor, gradOut: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_Negate")
	_Negate :: proc(ctx: ^Context, t: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_Log")
	_Log :: proc(ctx: ^Context, t: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_Abs")
	_Abs :: proc(ctx: ^Context, t: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_Sqrt")
	_Sqrt :: proc(ctx: ^Context, t: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_SqrtBackward")
	_SqrtBackward :: proc(ctx: ^Context, tensor: ^Raw_Tensor) ---

	// Reduction
	@(link_name = "shapes_Sum")
	_Sum :: proc(ctx: ^Context, t: ^Raw_Tensor, dim: uint) -> Raw_Tensor ---
	@(link_name = "shapes_ReduceBroadcast")
	_ReduceBroadcast :: proc(ctx: ^Context, input: ^Raw_Tensor, grad: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_Mean")
	_Mean :: proc(ctx: ^Context, t: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_MeanDim")
	_MeanDim :: proc(ctx: ^Context, t: ^Raw_Tensor, dim: uint) -> Raw_Tensor ---
	@(link_name = "shapes_Std")
	_Std :: proc(ctx: ^Context, t: ^Raw_Tensor) -> Raw_Tensor ---
	@(link_name = "shapes_Max")
	_Max :: proc(ctx: ^Context, t: ^Raw_Tensor, dim: uint) -> Raw_Tensor ---
	@(link_name = "shapes_ArgMax")
	_ArgMax :: proc(ctx: ^Context, t: ^Raw_Tensor, dim: uint) -> Raw_Tensor ---

	// Layer ops
	@(link_name = "shapes_DenseLinear")
	_DenseLinear :: proc(ctx: ^Context, x: ^Raw_Tensor, w: ^Raw_Tensor, bias: ^Raw_Tensor, withBias: bool) -> Raw_Tensor ---
	@(link_name = "shapes_DenseBackward")
	_DenseBackward :: proc(ctx: ^Context, x: ^Raw_Tensor, w: ^Raw_Tensor, gradOut: ^Raw_Tensor, dX: ^Raw_Tensor, dW: ^Raw_Tensor, dB: ^Raw_Tensor) -> Result ---
	@(link_name = "shapes_BatchNormForwardTraining")
	_BatchNormForwardTraining :: proc(ctx: ^Context, x2d: ^Raw_Tensor, gamma: ^Raw_Tensor, beta: ^Raw_Tensor, epsilon: f32) -> Raw_BatchNormForwardResult ---
	@(link_name = "shapes_BatchNormBackward")
	_BatchNormBackward :: proc(ctx: ^Context, x2d: ^Raw_Tensor, grad2d: ^Raw_Tensor, gamma: ^Raw_Tensor, epsilon: f32) -> Raw_BatchNormBackwardResult ---
	@(link_name = "shapes_Conv2d")
	_Conv2d :: proc(ctx: ^Context, inChannels: uint, outChannels: uint, stride: u8, kernels: ^Raw_Tensor, bias: ^Raw_Tensor, withBias: bool, t: ^Raw_Tensor, dest: ^Raw_Tensor, colBuffer: ^Raw_Tensor) -> Result ---
	@(link_name = "shapes_Conv2dBackward")
	_Conv2dBackward :: proc(ctx: ^Context, input: ^Raw_Tensor, dInput: ^Raw_Tensor, kernels: ^Raw_Tensor, dKernels: ^Raw_Tensor, outputGrad: ^Raw_Tensor, colBuffer: ^Raw_Tensor, dBias: ^Raw_Tensor, withBias: bool, stride: u8) -> Result ---
	@(link_name = "shapes_ConvTranspose2d")
	_ConvTranspose2d :: proc(ctx: ^Context, inChannels: uint, outChannels: uint, stride: u8, kernels: ^Raw_Tensor, kernel: Raw_Dim, t: ^Raw_Tensor, dest: ^Raw_Tensor) -> Result ---
	@(link_name = "shapes_ConvTranspose2dBackward")
	_ConvTranspose2dBackward :: proc(ctx: ^Context, x: ^Raw_Tensor, kernels: ^Raw_Tensor, gradOut: ^Raw_Tensor, stride: u8, dX: ^Raw_Tensor, dKernels: ^Raw_Tensor) -> Result ---
	@(link_name = "shapes_MaxPool2d")
	_MaxPool2d :: proc(ctx: ^Context, x: ^Raw_Tensor, kernel: Raw_Dim, stride: u8, dest: ^Raw_Tensor) -> Result ---
	@(link_name = "shapes_MaxPool2dWithIndices")
	_MaxPool2dWithIndices :: proc(ctx: ^Context, x: ^Raw_Tensor, kernel: Raw_Dim, stride: u8, dest: ^Raw_Tensor, indices: ^Raw_Tensor) -> Result ---
	@(link_name = "shapes_MaxPool2dBackward")
	_MaxPool2dBackward :: proc(ctx: ^Context, x: ^Raw_Tensor, gradOut: ^Raw_Tensor, kernel: Raw_Dim, stride: u8, dX: ^Raw_Tensor) -> Result ---
	@(link_name = "shapes_MaxPool2dBackwardWithIndices")
	_MaxPool2dBackwardWithIndices :: proc(ctx: ^Context, x: ^Raw_Tensor, gradOut: ^Raw_Tensor, indices: ^Raw_Tensor, dX: ^Raw_Tensor) -> Result ---
	@(link_name = "shapes_AdaptiveAvgPool2d")
	_AdaptiveAvgPool2d :: proc(ctx: ^Context, x: ^Raw_Tensor, outH: uint, outW: uint, dest: ^Raw_Tensor) -> Result ---
	@(link_name = "shapes_AdaptiveAvgPool2dBackward")
	_AdaptiveAvgPool2dBackward :: proc(ctx: ^Context, x: ^Raw_Tensor, gradOut: ^Raw_Tensor, outH: uint, outW: uint, dX: ^Raw_Tensor) -> Result ---

	// Loss
	@(link_name = "shapes_loss_CrossEntropyForward")
	_loss_CrossEntropyForward :: proc(ctx: ^Context, yGround: ^Raw_Tensor, logits: ^Raw_Tensor) -> Raw_TensorPair ---
	@(link_name = "shapes_loss_CrossEntropyBackward")
	_loss_CrossEntropyBackward :: proc(ctx: ^Context, yGround: ^Raw_Tensor, probs: ^Raw_Tensor, gradOut: ^Raw_Tensor) -> Raw_Tensor ---

	// Optimizer
	@(link_name = "shapes_optimizer_Sgd")
	_optimizer_Sgd :: proc(ctx: ^Context, parameters: ^olib.Array, learningRate: f32) -> Result ---
	@(link_name = "shapes_optimizer_Adam")
	_optimizer_Adam :: proc(ctx: ^Context, triplets: ^Raw_AdamData, numTriplets: uint, b1: f32, b2: f32, step: uint, a: f32, epsilon: f32) -> Result ---
}

Scalar :: proc() -> Dim {
	dims := make([]uint, 2)
	dims[0] = 1
	shape := Dim {
		dims = dims,
	}

	return shape
}

Shape1D :: proc(dim0: uint) -> Dim {
	dims := make([]uint, 2)
	dims[0] = dim0
	shape := Dim {
		dims = dims,
	}

	return shape
}

Shape2D :: proc(dim0: uint, dim1: uint) -> Dim {
	dims := make([]uint, 2)
	dims[0] = dim0
	dims[1] = dim1

	shape := Dim {
		dims = dims,
	}

	return shape
}

ToRawDim :: proc(dim: Dim) -> Raw_Dim {
	return Raw_Dim {
		dims = raw_data(dim.dims),
		multipliers = raw_data(dim.multipliers),
		numOfDims = u8(len(dim.dims)),
	}
}

PrintTensor :: proc(tensor: Tensor) {
	rawTensor := ToRawTensor(tensor)
	_PrintTensor(&rawTensor)
}

Size :: proc(dim: Dim) -> uint {
	size: uint = 1
	for d in dim.dims {
		size *= d
	}

	return size
}

DtypeFromTypeid :: proc($T: typeid) -> Dtype {
	when T == f16 {
		return .F16
	} else when T == f32 {
		return .F32
	} else when T == f64 {
		return .F64
	} else when T == u8 {
		return .U8
	} else when T == u16 {
		return .U16
	} else when T == u32 {
		return .U32
	} else when T == u64 {
		return .U64
	} else when T == i8 {
		return .I8
	} else when T == i16 {
		return .I16
	} else when T == i32 {
		return .I32
	} else when T == i64 {
		return .I64
	} else when T == bool {
		return .BOOL
	} else {
		return .F32
	}
}

TypeidFromDtype :: proc(dtype: Dtype) -> typeid {
	switch dtype {
	case .F16:
		return f16
	case .F32:
		return f32
	case .F64:
		return f64
	case .U8:
		return u8
	case .U16:
		return u16
	case .U32:
		return u32
	case .U64:
		return u64
	case .I8:
		return i8
	case .I16:
		return i16
	case .I32:
		return i32
	case .I64:
		return i64
	case .BOOL:
		return bool
	}
	return f32
}

FromRawTensor :: proc(t: Raw_Tensor) -> Tensor {
	values := mem.slice_ptr(cast(^TensorValue)t.values, int(t.size))
	t := Tensor {
		ctx = t.ctx,
		label = t.label,
		metadataMemory = t.metadataMemory,
		values = values,
		dtype = t.dtype,
		boundary = t.boundary,
		size = t.size,
		shape = Dim {
			dims = t.shape.dims[0:t.shape.numOfDims],
			multipliers = t.shape.multipliers[0:t.shape.numOfDims],
		},
		isView = t.isView,
		isContigous = t.isContigous,
		isContigousCopy = t.isContigousCopy,
		grad = (^Tensor)(t.grad),
		nodeId = t.nodeId,
	}

	return t
}

ToRawTensor :: proc(t: Tensor) -> Raw_Tensor {
	return Raw_Tensor {
		ctx = t.ctx,
		dtype = t.dtype,
		label = t.label,
		metadataMemory = t.metadataMemory,
		values = raw_data(t.values),
		boundary = t.boundary,
		size = t.size,
		shape = Raw_Dim {
			dims = raw_data(t.shape.dims),
			multipliers = raw_data(t.shape.multipliers),
			numOfDims = u8(len(t.shape.dims)),
		},
		isView = t.isView,
		isContigous = t.isContigous,
		isContigousCopy = t.isContigousCopy,
		grad = t.grad,
		nodeId = t.nodeId,
	}
}

Add :: proc(a, b: Tensor) -> Tensor {
	ctx := cast(^Context)context.user_ptr
	_a := ToRawTensor(a)
	_b := ToRawTensor(b)

	result := _Add(ctx, &_a, &_b)
	return FromRawTensor(result)
}

Multiply :: proc(a, b: Tensor) -> Tensor {
	ctx := cast(^Context)context.user_ptr
	_a := ToRawTensor(a)
	_b := ToRawTensor(b)

	result := _Multiply(ctx, &_a, &_b)
	return FromRawTensor(result)
}

MakeDynamicTensorArray :: proc() -> ArrayTensor {
	ctx := cast(^Context)context.user_ptr
	return _Make_DynamicTensorArray(ctx.memory)
}

MakeTensorArray :: proc(capacity: uint) -> ^olib.Array {
	ctx := cast(^Context)context.user_ptr
	return _MakeTensorArray(ctx.memory, capacity)
}

ArrayAppendTensor :: proc(array: ^olib.Array, tensor: ^Raw_Tensor) {
	_ArrayAppendTensor(array, tensor)
}

ArrayAppendTensorArray :: proc(array: ^olib.Array, tensorArray: ^olib.Array) {
	_ArrayAppendTensorArray(array, tensorArray)
}

// Tensor creation
MakeZeros :: proc(shape: Raw_Dim, $T: typeid) -> Tensor {
	ctx := cast(^Context)context.user_ptr
	t := _MakeZerosTensorWithType(ctx, ToRawDim(shape), T)
	tensor := FromRawTensor(t)
	return tensor
}

MakeFilled :: proc(shape: Dim, $T: typeid, initialValue: T) -> Tensor {
	ctx := cast(^Context)context.user_ptr

	dtype := DtypeFromTypeid(T)
	rawTensor := _MakeZerosTensorWithType(ctx, ToRawDim(shape), dtype)
	tensor := FromRawTensor(rawTensor)
	for vi in 0 ..< tensor.size {
		tensor.values[vi] = initialValue
	}

	return FromRawTensor(rawTensor)
}

MakeFromContigousArray :: proc(shape: Raw_Dim, values: rawptr, $T: typeid) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _MakeFromContigousArray(ctx, shape, values, DtypeFromTypeid(T))
}

MakeRandomTensor :: proc(minValue: f32, maxValue: f32, shape: Dim) -> Tensor {
	ctx := cast(^Context)context.user_ptr
	t := _MakeRandomTensor(ctx, ToRawDim(shape), minValue, maxValue, DtypeFromTypeid(f32))
	return FromRawTensor(t)
}

MakeOneHotTensor :: proc(indices: ^Raw_Tensor, numClasses: uint) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _MakeOneHotTensor(ctx, indices, numClasses)
}

MakeArangeTensor :: proc(start: f32, end: f32, step: f32) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _MakeArangeTensor(ctx, start, end, step)
}

SetValues :: proc(t: ^Raw_Tensor, value: Raw_Value) {
	_SetValues(t, value)
}

// Binary ops
Subtract :: proc(a: ^Raw_Tensor, b: ^Raw_Tensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _Subtract(ctx, a, b)
}

Divide :: proc(a: ^Raw_Tensor, b: ^Raw_Tensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _Divide(ctx, a, b)
}

GreaterThan :: proc(a: ^Raw_Tensor, b: ^Raw_Tensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _GreaterThan(ctx, a, b)
}

GreaterThanOrEqual :: proc(a: ^Raw_Tensor, b: ^Raw_Tensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _GreaterThanOrEqual(ctx, a, b)
}

Equal :: proc(a: ^Raw_Tensor, b: ^Raw_Tensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _Equal(ctx, a, b)
}

LessThan :: proc(a: ^Raw_Tensor, b: ^Raw_Tensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _LessThan(ctx, a, b)
}

LessThanOrEqual :: proc(a: ^Raw_Tensor, b: ^Raw_Tensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _LessThanOrEqual(ctx, a, b)
}

MatMul :: proc(a: ^Raw_Tensor, b: ^Raw_Tensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _MatMul(ctx, a, b)
}

Dot :: proc(a: ^Raw_Tensor, b: ^Raw_Tensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _Dot(ctx, a, b)
}

// Inplace ops
AddInPlace :: proc(a: ^Raw_Tensor, b: ^Raw_Tensor) {
	ctx := cast(^Context)context.user_ptr
	_AddInPlace(ctx, a, b)
}

SubtractInPlace :: proc(a: ^Raw_Tensor, b: ^Raw_Tensor) {
	ctx := cast(^Context)context.user_ptr
	_SubtractInPlace(ctx, a, b)
}

MultiplyInPlace :: proc(a: ^Raw_Tensor, b: ^Raw_Tensor) {
	ctx := cast(^Context)context.user_ptr
	_MultiplyInPlace(ctx, a, b)
}

// Indexing

GetAt :: proc(t: ^Raw_Tensor, dim: Raw_Dim) -> ^Raw_Value {
	return _GetAt(t, dim)
}

IndexWithTensor :: proc(source: ^Raw_Tensor, indices: ^Raw_Tensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _IndexWithTensor(ctx, source, indices)
}

IndexWithTensor2d :: proc(
	source: ^Raw_Tensor,
	rowIndices: ^Raw_Tensor,
	colIndices: ^Raw_Tensor,
) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _IndexWithTensor2d(ctx, source, rowIndices, colIndices)
}

IndexAccumulate1d :: proc(dest: ^Raw_Tensor, indices: ^Raw_Tensor, srcGrad: ^Raw_Tensor) {
	ctx := cast(^Context)context.user_ptr
	_IndexAccumulate1d(ctx, dest, indices, srcGrad)
}

IndexAccumulate2d :: proc(
	dest: ^Raw_Tensor,
	rowIndices: ^Raw_Tensor,
	colIndices: ^Raw_Tensor,
	srcGrad: ^Raw_Tensor,
) {
	ctx := cast(^Context)context.user_ptr
	_IndexAccumulate2d(ctx, dest, rowIndices, colIndices, srcGrad)
}

ReluBackwardAccumulate :: proc(output: ^Raw_Tensor, gradOut: ^Raw_Tensor, dest: ^Raw_Tensor) {
	ctx := cast(^Context)context.user_ptr
	_ReluBackwardAccumulate(ctx, output, gradOut, dest)
}

// Shape manipulation

Reshape :: proc(source: ^Raw_Tensor, dim: Raw_Dim) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _Reshape(ctx, source, dim)
}

ReshapeBackward :: proc(node: ^Raw_Tensor) {
	ctx := cast(^Context)context.user_ptr
	_ReshapeBackward(ctx, node)
}

Transpose :: proc(source: ^Raw_Tensor, dim0: u8, dim1: u8) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _Transpose(ctx, source, uint(dim0), uint(dim1))
}

Permute :: proc(source: ^Raw_Tensor, order: Raw_Dim) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _Permute(ctx, source, order)
}

Squeeze :: proc(t: ^Raw_Tensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _Squeeze(ctx, t)
}

SqueezeDim :: proc(t: ^Raw_Tensor, dim: uint) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _SqueezeDim(ctx, t, dim)
}

UnSqueeze :: proc(t: ^Raw_Tensor, dim: uint) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _UnSqueeze(ctx, t, dim)
}

// Copying

Clone :: proc(tensor: ^Raw_Tensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _Clone(ctx, tensor)
}

Copy :: proc(src: ^Raw_Tensor, dest: ^Raw_Tensor) {
	ctx := cast(^Context)context.user_ptr
	_Copy(ctx, src, dest)
}

Concat :: proc(target: ^Raw_Tensor, targetDim: uint, tensors: ArrayTensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _Concat(ctx, target, targetDim, tensors)
}

Stack :: proc(tensors: ArrayTensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _Stack(ctx, tensors)
}

// Unary

Cast :: proc(source: ^Raw_Tensor, $T: typeid) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _Cast(ctx, source, DtypeFromTypeid(T))
}

Pow :: proc(t: ^Raw_Tensor, power: f32) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _Pow(ctx, t, power)
}

Exp :: proc(t: ^Raw_Tensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _Exp(ctx, t)
}

Tanh :: proc(t: ^Raw_Tensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _Tanh(ctx, t)
}

Relu :: proc(t: ^Raw_Tensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _Relu(ctx, t)
}

ReluBackward :: proc(output: ^Raw_Tensor, gradOut: ^Raw_Tensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _ReluBackward(ctx, output, gradOut)
}

Negate :: proc(t: ^Raw_Tensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _Negate(ctx, t)
}

Log :: proc(t: ^Raw_Tensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _Log(ctx, t)
}

Abs :: proc(t: ^Raw_Tensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _Abs(ctx, t)
}

Sqrt :: proc(t: ^Raw_Tensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _Sqrt(ctx, t)
}

SqrtBackward :: proc(tensor: ^Raw_Tensor) {
	ctx := cast(^Context)context.user_ptr
	_SqrtBackward(ctx, tensor)
}

// Reduction

Sum :: proc(t: ^Raw_Tensor, dim: uint) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _Sum(ctx, t, dim)
}

ReduceBroadcast :: proc(input: ^Raw_Tensor, grad: ^Raw_Tensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _ReduceBroadcast(ctx, input, grad)
}

Mean :: proc(t: ^Raw_Tensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _Mean(ctx, t)
}

MeanDim :: proc(t: ^Raw_Tensor, dim: uint) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _MeanDim(ctx, t, dim)
}

Std :: proc(t: ^Raw_Tensor) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _Std(ctx, t)
}

Max :: proc(t: ^Raw_Tensor, dim: uint) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _Max(ctx, t, dim)
}

ArgMax :: proc(t: ^Raw_Tensor, dim: uint) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _ArgMax(ctx, t, dim)
}

// Layer ops


BatchNormForwardTraining :: proc(
	x2d: ^Raw_Tensor,
	gamma: ^Raw_Tensor,
	beta: ^Raw_Tensor,
	epsilon: f32,
) -> Raw_BatchNormForwardResult {
	ctx := cast(^Context)context.user_ptr
	return _BatchNormForwardTraining(ctx, x2d, gamma, beta, epsilon)
}

BatchNormBackward :: proc(
	x2d: ^Raw_Tensor,
	grad2d: ^Raw_Tensor,
	gamma: ^Raw_Tensor,
	epsilon: f32,
) -> Raw_BatchNormBackwardResult {
	ctx := cast(^Context)context.user_ptr
	return _BatchNormBackward(ctx, x2d, grad2d, gamma, epsilon)
}

Conv2d :: proc(
	inChannels: uint,
	outChannels: uint,
	stride: u8,
	kernels: ^Raw_Tensor,
	bias: ^Raw_Tensor,
	withBias: bool,
	t: ^Raw_Tensor,
	dest: ^Raw_Tensor,
	colBuffer: ^Raw_Tensor,
) -> Result {
	ctx := cast(^Context)context.user_ptr
	return _Conv2d(
		ctx,
		inChannels,
		outChannels,
		stride,
		kernels,
		bias,
		withBias,
		t,
		dest,
		colBuffer,
	)
}

Conv2dBackward :: proc(
	input: ^Raw_Tensor,
	dInput: ^Raw_Tensor,
	kernels: ^Raw_Tensor,
	dKernels: ^Raw_Tensor,
	outputGrad: ^Raw_Tensor,
	colBuffer: ^Raw_Tensor,
	dBias: ^Raw_Tensor,
	withBias: bool,
	stride: u8,
) -> Result {
	ctx := cast(^Context)context.user_ptr
	return _Conv2dBackward(
		ctx,
		input,
		dInput,
		kernels,
		dKernels,
		outputGrad,
		colBuffer,
		dBias,
		withBias,
		stride,
	)
}

ConvTranspose2d :: proc(
	inChannels: uint,
	outChannels: uint,
	stride: u8,
	kernels: ^Raw_Tensor,
	kernel: Raw_Dim,
	t: ^Raw_Tensor,
	dest: ^Raw_Tensor,
) -> Result {
	ctx := cast(^Context)context.user_ptr
	return _ConvTranspose2d(ctx, inChannels, outChannels, stride, kernels, kernel, t, dest)
}

ConvTranspose2dBackward :: proc(
	x: ^Raw_Tensor,
	kernels: ^Raw_Tensor,
	gradOut: ^Raw_Tensor,
	stride: u8,
	dX: ^Raw_Tensor,
	dKernels: ^Raw_Tensor,
) -> Result {
	ctx := cast(^Context)context.user_ptr
	return _ConvTranspose2dBackward(ctx, x, kernels, gradOut, stride, dX, dKernels)
}

MaxPool2d :: proc(x: ^Raw_Tensor, kernel: Raw_Dim, stride: u8, dest: ^Raw_Tensor) -> Result {
	ctx := cast(^Context)context.user_ptr
	return _MaxPool2d(ctx, x, kernel, stride, dest)
}

MaxPool2dWithIndices :: proc(
	x: ^Raw_Tensor,
	kernel: Raw_Dim,
	stride: u8,
	dest: ^Raw_Tensor,
	indices: ^Raw_Tensor,
) -> Result {
	ctx := cast(^Context)context.user_ptr
	return _MaxPool2dWithIndices(ctx, x, kernel, stride, dest, indices)
}

MaxPool2dBackward :: proc(
	x: ^Raw_Tensor,
	gradOut: ^Raw_Tensor,
	kernel: Raw_Dim,
	stride: u8,
	dX: ^Raw_Tensor,
) -> Result {
	ctx := cast(^Context)context.user_ptr
	return _MaxPool2dBackward(ctx, x, gradOut, kernel, stride, dX)
}

MaxPool2dBackwardWithIndices :: proc(
	x: ^Raw_Tensor,
	gradOut: ^Raw_Tensor,
	indices: ^Raw_Tensor,
	dX: ^Raw_Tensor,
) -> Result {
	ctx := cast(^Context)context.user_ptr
	return _MaxPool2dBackwardWithIndices(ctx, x, gradOut, indices, dX)
}

AdaptiveAvgPool2d :: proc(x: ^Raw_Tensor, outH: uint, outW: uint, dest: ^Raw_Tensor) -> Result {
	ctx := cast(^Context)context.user_ptr
	return _AdaptiveAvgPool2d(ctx, x, outH, outW, dest)
}

AdaptiveAvgPool2dBackward :: proc(
	x: ^Raw_Tensor,
	gradOut: ^Raw_Tensor,
	outH: uint,
	outW: uint,
	dX: ^Raw_Tensor,
) -> Result {
	ctx := cast(^Context)context.user_ptr
	return _AdaptiveAvgPool2dBackward(ctx, x, gradOut, outH, outW, dX)
}

// Loss

LossCrossEntropyForward :: proc(yGround: ^Raw_Tensor, logits: ^Raw_Tensor) -> Raw_TensorPair {
	ctx := cast(^Context)context.user_ptr
	return _loss_CrossEntropyForward(ctx, yGround, logits)
}

LossCrossEntropyBackward :: proc(
	yGround: ^Raw_Tensor,
	probs: ^Raw_Tensor,
	gradOut: ^Raw_Tensor,
) -> Raw_Tensor {
	ctx := cast(^Context)context.user_ptr
	return _loss_CrossEntropyBackward(ctx, yGround, probs, gradOut)
}

// Optimizer

OptimizerSgd :: proc(parameters: ^olib.Array, learningRate: f32) -> Result {
	ctx := cast(^Context)context.user_ptr
	return _optimizer_Sgd(ctx, parameters, learningRate)
}

OptimizerAdam :: proc(
	triplets: ^Raw_AdamData,
	numTriplets: uint,
	b1: f32,
	b2: f32,
	step: uint,
	a: f32,
	epsilon: f32,
) -> Result {
	ctx := cast(^Context)context.user_ptr
	return _optimizer_Adam(ctx, triplets, numTriplets, b1, b2, step, a, epsilon)
}

