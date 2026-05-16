#ifndef TEST_H
#define TEST_H

#include <stdio.h>
#include <stdlib.h>

extern int tests_passed;
extern int tests_failed;

#define ASSERT(condition, msg)                                                                                                                       \
  do {                                                                                                                                               \
    if (condition) {                                                                                                                                 \
      tests_passed++;                                                                                                                                \
    } else {                                                                                                                                         \
      tests_failed++;                                                                                                                                \
      printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);                                                                                         \
    }                                                                                                                                                \
  } while (0)

#define ASSERT_EQ(a, b, msg)      ASSERT((a) == (b), msg)
#define ASSERT_NEQ(a, b, msg)     ASSERT((a) != (b), msg)
#define ASSERT_NULL(ptr, msg)     ASSERT((ptr) == NULL, msg)
#define ASSERT_NOT_NULL(ptr, msg) ASSERT((ptr) != NULL, msg)

#define TEST_SUMMARY()                                                                                                                               \
  printf("\n%d passed, %d failed\n", tests_passed, tests_failed);                                                                                    \
  return tests_failed > 0 ? 1 : 0;

#endif
