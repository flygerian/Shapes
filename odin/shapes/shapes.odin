package shapes

import "core:c"
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

Dim :: struct {
	dims:        [^]uint,
	multipliers: [^]uint,
	numOfDims:   u8,
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

Value :: struct {
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

BatchNormForwardResult :: struct {
	out:      Tensor,
	mean:     Tensor,
	variance: Tensor,
}

BatchNormBackwardResult :: struct {
	dx2d:   Tensor,
	dGamma: Tensor,
	dBeta:  Tensor,
}

TensorPair :: struct {
	a: Tensor,
	b: Tensor,
}

AdamData :: struct {
	param: rawptr,
	grad:  rawptr,
	m:     rawptr,
	v:     rawptr,
	size:  uint,
	dtype: Dtype,
}

Tensor :: struct {
	ctx:             ^Context,
	label:           cstring,
	metadataMemory:  ^olib.Memory,
	values:          rawptr,
	boundary:        ^Range,
	size:            uint,
	shape:           Dim,
	dtype:           Dtype,
	isView:          bool,
	isContigous:     bool,
	isContigousCopy: bool,
	grad:            ^Tensor,
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
	GetScratchContext :: proc(ctx: ^Context, bufferSize: uint) -> Context ---
	DestroyContext :: proc(ctx: ^Context) ---
	FreeContext :: proc(ctx: ^Context) ---
	Flush :: proc(ctx: ^Context) -> Result ---
	CopyBetweenDevices :: proc(srcType: DeviceType, destType: DeviceType, srcPtr: rawptr, destPtr: rawptr, size: uint) -> Result ---
	MoveToCuda :: proc(destCtx: ^Context, tensors: ArrayTensor) ---
	MoveToHost :: proc(destCtx: ^Context, tensors: ArrayTensor) ---
	MoveTensorToHost :: proc(destCtx: ^Context, t: ^Tensor) ---

	// Array helpers
	Make_DynamicTensorArray :: proc(memory: ^olib.Memory) -> ArrayTensor ---
	MakeTensorArray :: proc(memory: ^olib.Memory, capacity: uint) -> ^olib.Array ---
	ArrayAppendTensor :: proc(array: ^olib.Array, tensor: ^Tensor) ---
	ArrayAppendTensorArray :: proc(array: ^olib.Array, tensorArray: ^olib.Array) ---

	// Tensor creation
	MakeZerosTensor :: proc(ctx: ^Context, shape: Dim) -> Tensor ---
	MakeIntTensor :: proc(ctx: ^Context, shape: Dim, initialValue: i8) -> Tensor ---
	MakeUIntTensor :: proc(ctx: ^Context, shape: Dim, initialValue: u8) -> Tensor ---
	MakeFloatTensor :: proc(ctx: ^Context, shape: Dim, initialValue: f32) -> Tensor ---
	MakeFloat64Tensor :: proc(ctx: ^Context, shape: Dim, initialValue: f64) -> Tensor ---
	MakeFromContigousArray :: proc(ctx: ^Context, shape: Dim, values: rawptr, dtype: Dtype) -> Tensor ---
	MakeRandomTensor :: proc(ctx: ^Context, shape: Dim, minValue: f32, maxValue: f32, dtype: Dtype) -> Tensor ---
	MakeOneHotTensor :: proc(ctx: ^Context, indices: ^Tensor, numClasses: uint) -> Tensor ---
	MakeArangeTensor :: proc(ctx: ^Context, start: f32, end: f32, step: f32) -> Tensor ---
	SetValues :: proc(t: ^Tensor, value: Value) ---

	// Binary ops
	Add :: proc(ctx: ^Context, a: ^Tensor, b: ^Tensor) -> Tensor ---
	Subtract :: proc(ctx: ^Context, a: ^Tensor, b: ^Tensor) -> Tensor ---
	Divide :: proc(ctx: ^Context, a: ^Tensor, b: ^Tensor) -> Tensor ---
	Multiply :: proc(ctx: ^Context, a: ^Tensor, b: ^Tensor) -> Tensor ---
	GreaterThan :: proc(ctx: ^Context, a: ^Tensor, b: ^Tensor) -> Tensor ---
	GreaterThanOrEqual :: proc(ctx: ^Context, a: ^Tensor, b: ^Tensor) -> Tensor ---
	Equal :: proc(ctx: ^Context, a: ^Tensor, b: ^Tensor) -> Tensor ---
	LessThan :: proc(ctx: ^Context, a: ^Tensor, b: ^Tensor) -> Tensor ---
	LessThanOrEqual :: proc(ctx: ^Context, a: ^Tensor, b: ^Tensor) -> Tensor ---
	MatMul :: proc(ctx: ^Context, a: ^Tensor, b: ^Tensor) -> Tensor ---
	Dot :: proc(ctx: ^Context, a: ^Tensor, b: ^Tensor) -> Tensor ---

	// Inplace ops
	AddInPlace :: proc(ctx: ^Context, a: ^Tensor, b: ^Tensor) ---
	SubtractInPlace :: proc(ctx: ^Context, a: ^Tensor, b: ^Tensor) ---
	MultiplyInPlace :: proc(ctx: ^Context, a: ^Tensor, b: ^Tensor) ---

	// Indexing
	GetAt :: proc(t: ^Tensor, dim: Dim) -> ^Value ---
	IndexWithTensor :: proc(ctx: ^Context, source: ^Tensor, indices: ^Tensor) -> Tensor ---
	IndexWithTensor2d :: proc(ctx: ^Context, source: ^Tensor, rowIndices: ^Tensor, colIndices: ^Tensor) -> Tensor ---
	IndexAccumulate1d :: proc(ctx: ^Context, dest: ^Tensor, indices: ^Tensor, srcGrad: ^Tensor) ---
	IndexAccumulate2d :: proc(ctx: ^Context, dest: ^Tensor, rowIndices: ^Tensor, colIndices: ^Tensor, srcGrad: ^Tensor) ---
	ReluBackwardAccumulate :: proc(ctx: ^Context, output: ^Tensor, gradOut: ^Tensor, dest: ^Tensor) ---

	// Shape manipulation
	Reshape :: proc(ctx: ^Context, source: ^Tensor, dim: Dim) -> Tensor ---
	ReshapeBackward :: proc(ctx: ^Context, node: ^Tensor) ---
	Transpose :: proc(ctx: ^Context, source: ^Tensor, #c_vararg args: ..any) -> Tensor ---
	Permute :: proc(ctx: ^Context, source: ^Tensor, order: Dim) -> Tensor ---
	Squeeze :: proc(ctx: ^Context, t: ^Tensor) -> Tensor ---
	SqueezeDim :: proc(ctx: ^Context, t: ^Tensor, dim: uint) -> Tensor ---
	UnSqueeze :: proc(ctx: ^Context, t: ^Tensor, dim: uint) -> Tensor ---

	// Copying
	Clone :: proc(ctx: ^Context, tensor: ^Tensor) -> Tensor ---
	Copy :: proc(ctx: ^Context, src: ^Tensor, dest: ^Tensor) ---
	Concat :: proc(ctx: ^Context, target: ^Tensor, targetDim: uint, tensors: ArrayTensor) -> Tensor ---
	Stack :: proc(ctx: ^Context, tensors: ArrayTensor) -> Tensor ---

	// Unary
	@(link_name = "Cast")
	Cast :: proc(ctx: ^Context, source: ^Tensor, targetDtype: Dtype) -> Tensor ---
	Pow :: proc(ctx: ^Context, t: ^Tensor, power: f32) -> Tensor ---
	Exp :: proc(ctx: ^Context, t: ^Tensor) -> Tensor ---
	Tanh :: proc(ctx: ^Context, t: ^Tensor) -> Tensor ---
	Relu :: proc(ctx: ^Context, t: ^Tensor) -> Tensor ---
	ReluBackward :: proc(ctx: ^Context, output: ^Tensor, gradOut: ^Tensor) -> Tensor ---
	Negate :: proc(ctx: ^Context, t: ^Tensor) -> Tensor ---
	Log :: proc(ctx: ^Context, t: ^Tensor) -> Tensor ---
	Abs :: proc(ctx: ^Context, t: ^Tensor) -> Tensor ---
	Sqrt :: proc(ctx: ^Context, t: ^Tensor) -> Tensor ---
	SqrtBackward :: proc(ctx: ^Context, tensor: ^Tensor) ---

	// Reduction
	Sum :: proc(ctx: ^Context, t: ^Tensor, dim: uint) -> Tensor ---
	ReduceBroadcast :: proc(ctx: ^Context, input: ^Tensor, grad: ^Tensor) -> Tensor ---
	Mean :: proc(ctx: ^Context, t: ^Tensor) -> Tensor ---
	MeanDim :: proc(ctx: ^Context, t: ^Tensor, dim: uint) -> Tensor ---
	Std :: proc(ctx: ^Context, t: ^Tensor) -> Tensor ---
	Max :: proc(ctx: ^Context, t: ^Tensor, dim: uint) -> Tensor ---
	ArgMax :: proc(ctx: ^Context, t: ^Tensor, dim: uint) -> Tensor ---

	// Layer ops
	DenseLinear :: proc(ctx: ^Context, x: ^Tensor, w: ^Tensor, bias: ^Tensor, withBias: bool) -> Tensor ---
	DenseBackward :: proc(ctx: ^Context, x: ^Tensor, w: ^Tensor, gradOut: ^Tensor, dX: ^Tensor, dW: ^Tensor, dB: ^Tensor) -> Result ---
	BatchNormForwardTraining :: proc(ctx: ^Context, x2d: ^Tensor, gamma: ^Tensor, beta: ^Tensor, epsilon: f32) -> BatchNormForwardResult ---
	BatchNormBackward :: proc(ctx: ^Context, x2d: ^Tensor, grad2d: ^Tensor, gamma: ^Tensor, epsilon: f32) -> BatchNormBackwardResult ---
	Conv2d :: proc(ctx: ^Context, inChannels: uint, outChannels: uint, stride: u8, kernels: ^Tensor, bias: ^Tensor, withBias: bool, t: ^Tensor, dest: ^Tensor, colBuffer: ^Tensor) -> Result ---
	Conv2dBackward :: proc(ctx: ^Context, input: ^Tensor, dInput: ^Tensor, kernels: ^Tensor, dKernels: ^Tensor, outputGrad: ^Tensor, colBuffer: ^Tensor, dBias: ^Tensor, withBias: bool, stride: u8) -> Result ---
	ConvTranspose2d :: proc(ctx: ^Context, inChannels: uint, outChannels: uint, stride: u8, kernels: ^Tensor, kernel: Dim, t: ^Tensor, dest: ^Tensor) -> Result ---
	ConvTranspose2dBackward :: proc(ctx: ^Context, x: ^Tensor, kernels: ^Tensor, gradOut: ^Tensor, stride: u8, dX: ^Tensor, dKernels: ^Tensor) -> Result ---
	MaxPool2d :: proc(ctx: ^Context, x: ^Tensor, kernel: Dim, stride: u8, dest: ^Tensor) -> Result ---
	MaxPool2dWithIndices :: proc(ctx: ^Context, x: ^Tensor, kernel: Dim, stride: u8, dest: ^Tensor, indices: ^Tensor) -> Result ---
	MaxPool2dBackward :: proc(ctx: ^Context, x: ^Tensor, gradOut: ^Tensor, kernel: Dim, stride: u8, dX: ^Tensor) -> Result ---
	MaxPool2dBackwardWithIndices :: proc(ctx: ^Context, x: ^Tensor, gradOut: ^Tensor, indices: ^Tensor, dX: ^Tensor) -> Result ---
	AdaptiveAvgPool2d :: proc(ctx: ^Context, x: ^Tensor, outH: uint, outW: uint, dest: ^Tensor) -> Result ---
	AdaptiveAvgPool2dBackward :: proc(ctx: ^Context, x: ^Tensor, gradOut: ^Tensor, outH: uint, outW: uint, dX: ^Tensor) -> Result ---

	// Loss
	loss_CrossEntropyForward :: proc(ctx: ^Context, yGround: ^Tensor, logits: ^Tensor) -> TensorPair ---
	loss_CrossEntropyBackward :: proc(ctx: ^Context, yGround: ^Tensor, probs: ^Tensor, gradOut: ^Tensor) -> Tensor ---

	// Optimizer
	optimizer_Sgd :: proc(ctx: ^Context, parameters: ^olib.Array, learningRate: f32) -> Result ---
	optimizer_Adam :: proc(ctx: ^Context, triplets: ^AdamData, numTriplets: uint, b1: f32, b2: f32, step: uint, a: f32, epsilon: f32) -> Result ---

	// Debug
	PrintItem :: proc(t: ^Tensor) ---
	GetItem :: proc(ctx: ^Context, t: ^Tensor) -> cstring ---
	PrintTensor :: proc(tensor: ^Tensor) ---
}

Shape1D :: proc(dim0: uint) -> Dim {
	dims := make([]uint, 2)
	dims[0] = dim0
	shape := Dim {
		dims      = raw_data(dims),
		numOfDims = 1,
	}

	return shape
}

Shape2D :: proc(dim0: uint, dim1: uint) -> Dim {
	dims := make([]uint, 2)
	dims[0] = dim0
	dims[1] = dim1

	shape := Dim {
		dims      = raw_data(dims),
		numOfDims = 2,
	}

	return shape
}

