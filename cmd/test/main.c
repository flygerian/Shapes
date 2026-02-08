#include "test.h"
#include "memory_test.h"
#include "tensor_test.h"
#include "activation_test.h"
#include "grad_test.h"

int tests_passed = 0;
int tests_failed = 0;

int main() {
  run_memory_tests();
  run_tensor_tests();
  run_activation_tests();
  run_grad_tests();
  TEST_SUMMARY();
}
