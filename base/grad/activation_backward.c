#include "common.h"
#include "grad/grad.h"
#include "result/result.h"
#include "tensor/tensor_internal.h"
#include "tensor/value.h"

Result tanhBackwards(Context *ctx, GraphNode *node) {
  Tensor *input = node->inputs[0];
  Tensor *grad_output = node->grad;
  Tensor *output = node->output;

  // Derivative of tanh: d(tanh(x))/dx = 1 - tanh^2(x)
  // grad_input = grad_output * (1 - output^2)

  // If input has no computation graph, no need to backprop further
  if (input->computation == NULL) {
    return OK;
  }

  Tensor *grad_input = input->computation->grad;

  // Element-wise: grad_input[i] += grad_output[i] * (1 - output[i]^2)
  for (size_t i = 0; i < output->size; i++) {
    Value out_val, grad_out_val, grad_in_val;

    VALUE_GET_FROM_ARR(output->values, i, &out_val, output->dtype);
    VALUE_GET_FROM_ARR(grad_output->values, i, &grad_out_val, grad_output->dtype);
    VALUE_GET_FROM_ARR(grad_input->values, i, &grad_in_val, grad_input->dtype);

    // Compute: grad_output * (1 - output^2)
    switch (output->dtype) {
      case F16: {
        f16 out = out_val.as.f16;
        f16 local_grad = 1.0f - out * out;
        grad_in_val.as.f16 += grad_out_val.as.f16 * local_grad;
        break;
      }
      case F32: {
        f32 out = out_val.as.f32;
        f32 local_grad = 1.0f - out * out;
        grad_in_val.as.f32 += grad_out_val.as.f32 * local_grad;
        break;
      }
      case F64: {
        f64 out = out_val.as.f64;
        f64 local_grad = 1.0 - out * out;
        grad_in_val.as.f64 += grad_out_val.as.f64 * local_grad;
        break;
      }
      default: return ERR_TANH_VALUE_NOT_FLOAT;
    }

    VALUE_SET(grad_input->values, i, grad_in_val);
  }

  return OK;
}

Result ConstructTanhBackwardpass(Context *ctx, Tensor *t, Tensor *result) {

  GraphNode *node = allocate(ctx->memory, sizeof(GraphNode));

  node->optype = OP_TANH;
  node->output = result;
  node->grad = t_Zeros(ctx, result->shape, result->dtype);
  node->inputs = (Tensor **)allocate(ctx->memory, sizeof(Tensor *) * 1);
  node->numInputs = 1;
  node->metadata = NULL;

  node->inputs[0] = t;
  node->backward = tanhBackwards;

  result->computation = node;

  return OK;
}

Result powBackwards(Context *ctx, GraphNode *node) {
  Tensor *input = node->inputs[0];
  Tensor *grad_output = node->grad;

  // Retrieve the power value from metadata
  f32 power = *((f32 *)node->metadata);

  // Derivative of x^n: d(x^n)/dx = n * x^(n-1)
  // grad_input = grad_output * power * input^(power-1)

  // If input has no computation graph, no need to backprop further
  if (input->computation == NULL) {
    return OK;
  }

  Tensor *grad_input = input->computation->grad;

  // Element-wise: grad_input[i] += grad_output[i] * power * input[i]^(power-1)
  for (size_t i = 0; i < input->size; i++) {
    Value in_val, grad_out_val, grad_in_val;

    VALUE_GET_FROM_ARR(input->values, i, &in_val, input->dtype);
    VALUE_GET_FROM_ARR(grad_output->values, i, &grad_out_val, grad_output->dtype);
    VALUE_GET_FROM_ARR(grad_input->values, i, &grad_in_val, grad_input->dtype);

    // Compute: grad_output * power * input^(power-1)
    switch (input->dtype) {
      case F16: {
        f16 x = in_val.as.f16;
        f16 local_grad = power * powf(x, power - 1.0f);
        grad_in_val.as.f16 += grad_out_val.as.f16 * local_grad;
        break;
      }
      case F32: {
        f32 x = in_val.as.f32;
        f32 local_grad = power * powf(x, power - 1.0f);
        grad_in_val.as.f32 += grad_out_val.as.f32 * local_grad;
        break;
      }
      case F64: {
        f64 x = in_val.as.f64;
        f64 local_grad = (f64)power * pow(x, (f64)power - 1.0);
        grad_in_val.as.f64 += grad_out_val.as.f64 * local_grad;
        break;
      }
      default: return ERR_POW_VALUE_NOT_FLOAT;
    }

    VALUE_SET(grad_input->values, i, grad_in_val);
  }

  return OK;
}

Result ConstructPowBackwardpass(Context *ctx, Tensor *t, f32 power, Tensor *result) {

  GraphNode *node = allocate(ctx->memory, sizeof(GraphNode));

  node->optype = OP_POW;
  node->output = result;
  node->grad = t_Zeros(ctx, result->shape, result->dtype);
  node->inputs = (Tensor **)allocate(ctx->memory, sizeof(Tensor *) * 1);
  node->numInputs = 1;

  // Allocate and store the power value in metadata
  f32 *power_ptr = (f32 *)allocate(ctx->memory, sizeof(f32));
  *power_ptr = power;
  node->metadata = (void *)power_ptr;

  node->inputs[0] = t;
  node->backward = powBackwards;

  result->computation = node;

  return OK;
}
