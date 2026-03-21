#include "../common.h"
#include "tensor_internal.h"
#include <sched.h>
#include <stdlib.h>
#include <string.h>

static const char *dtypeName(Dtype dtype) {
  switch (dtype) {
    case F16: return "F16";
    case F32: return "F32";
    case F64: return "F64";
    case U8: return "U8";
    case U16: return "U16";
    case U32: return "U32";
    case U64: return "U64";
    case I8: return "I8";
    case I16: return "I16";
    case I32: return "I32";
    case I64: return "I64";
    case BOOL: return "BOOL";
    default: return "UNKNOWN";
  }
}

static const char *deviceTypeName(DeviceType type) {
  switch (type) {
    case CPU: return "CPU";
    case CUDA: return "CUDA";
    default: return "UNKNOWN";
  }
}

static bool shouldLogGemm(void) {
  const char *value = getenv("SHAPES_LOG_GEMM");
  return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

static void logGemmDispatch(Context *ctx, Dtype dtype, CBLAS_TRANSPOSE transA,
                            CBLAS_TRANSPOSE transB, int m, int n, int k, bool accumulate) {
  if (!shouldLogGemm()) {
    return;
  }

  const char *deviceName = ctx->device == NULL ? "CPU(default)" : deviceTypeName(ctx->device->type);
  fprintf(stderr, "[runGemm] device=%s dtype=%s transA=%d transB=%d m=%d n=%d k=%d accumulate=%d\n",
          deviceName, dtypeName(dtype), (int)transA, (int)transB, m, n, k, accumulate ? 1 : 0);
}

static cublasOperation_t toCudaTranspose(CBLAS_TRANSPOSE trans) {
  switch (trans) {
    case CblasNoTrans: return CUBLAS_OP_N;
    case CblasTrans: return CUBLAS_OP_T;
    default: return CUBLAS_OP_C;
  }
}

void runCpuGemm(Dtype dtype, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int m, int n, int k,
                const void *a, int lda, const void *b, int ldb, bool accumulate, void *c, int ldc) {

  if (dtype == F64) {
    cblas_dgemm(CblasRowMajor, transA, transB, m, n, k, 1.0, (const double *)a, lda,
                (const double *)b, ldb, accumulate ? 1.0 : 0.0, (double *)c, ldc);
    return;
  }

  cblas_sgemm(CblasRowMajor, transA, transB, m, n, k, 1.0f, (const float *)a, lda, (const float *)b,
              ldb, accumulate ? 1.0f : 0.0f, (float *)c, ldc);
}

void runGemm(Context *ctx, Dtype dtype, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int m,
             int n, int k, const void *a, int lda, const void *b, int ldb, bool accumulate, void *c,
             int ldc) {
  logGemmDispatch(ctx, dtype, transA, transB, m, n, k, accumulate);

  if (ctx->device == NULL) {
    return runCpuGemm(dtype, transA, transB, m, n, k, a, lda, b, ldb, accumulate, c, ldc);
  }

  switch (ctx->device->type) {
    case CPU:
      return runCpuGemm(dtype, transA, transB, m, n, k, a, lda, b, ldb, accumulate, c, ldc);
      break;

    case CUDA:

      // cuBLAS assumes column-major storage; swapping operands/op flags makes it
      // compute the same result as the row-major CBLAS entry points used elsewhere.
      return runCudaGemm(ctx, dtype, toCudaTranspose(transB), toCudaTranspose(transA), n, m, k, b,
                         ldb, a, lda, accumulate, c, ldc);

    default: return runCpuGemm(dtype, transA, transB, m, n, k, a, lda, b, ldb, accumulate, c, ldc);
  }
}
