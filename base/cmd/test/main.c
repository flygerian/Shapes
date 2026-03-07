#include "test.h"
#include "memory_test.h"
#include "tensor_test.h"
#include "unary_test.h"
#include "cast_test.h"
#include "sgd_test.h"
#include "layer_test.h"

int tests_passed = 0;
int tests_failed = 0;

int main() {
  run_memory_tests();
  run_tensor_tests();
  run_unary_tests();
  run_cast_tests();
  run_sgd_tests();
  run_layer_tests();
  TEST_SUMMARY();
}
