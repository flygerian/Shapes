#include "test.h"
#include "../../memory.h"
#include <string.h>

static void test_allocate_returns_non_null(void) {
  Memory *mem = initializeMemory();
  void *ptr = allocate(mem, 64);
  ASSERT_NOT_NULL(ptr, "allocate should return non-null for valid size");
  freeMemory(mem);
}

static void test_allocate_zero_returns_null(void) {
  Memory *mem = initializeMemory();
  void *ptr = allocate(mem, 0);
  ASSERT_NULL(ptr, "allocate(0) should return NULL");
  freeMemory(mem);
}

static void test_allocate_multiple_blocks(void) {
  Memory *mem = initializeMemory();
  void *ptr1 = allocate(mem, 32);
  void *ptr2 = allocate(mem, 64);
  void *ptr3 = allocate(mem, 16);

  ASSERT_NOT_NULL(ptr1, "first allocation should succeed");
  ASSERT_NOT_NULL(ptr2, "second allocation should succeed");
  ASSERT_NOT_NULL(ptr3, "third allocation should succeed");
  ASSERT_NEQ(ptr1, ptr2, "allocations should return different addresses");
  ASSERT_NEQ(ptr2, ptr3, "allocations should return different addresses");

  freeMemory(mem);
}

static void test_allocate_data_integrity(void) {
  Memory *mem = initializeMemory();
  char *str = allocate(mem, 16);
  strcpy(str, "hello");
  ASSERT_EQ(strcmp(str, "hello"), 0, "allocated memory should hold data");
  freeMemory(mem);
}

static void test_free_and_reuse(void) {
  Memory *mem = initializeMemory();
  void *ptr1 = allocate(mem, 64);
  freeAlloc(mem, ptr1);
  void *ptr2 = allocate(mem, 32);
  ASSERT_NOT_NULL(ptr2, "should allocate after free");
  freeMemory(mem);
}

static void test_reallocate_grows_block(void) {
  Memory *mem = initializeMemory();
  char *ptr = allocate(mem, 16);
  strcpy(ptr, "test");
  ptr = reallocate(mem, ptr, 64);
  ASSERT_NOT_NULL(ptr, "reallocate should return non-null");
  ASSERT_EQ(strcmp(ptr, "test"), 0, "reallocate should preserve data");
  freeMemory(mem);
}

static void test_reallocate_null_allocates(void) {
  Memory *mem = initializeMemory();
  void *ptr = reallocate(mem, NULL, 32);
  ASSERT_NOT_NULL(ptr, "reallocate(NULL, size) should allocate");
  freeMemory(mem);
}

static void test_reallocate_to_zero_frees(void) {
  Memory *mem = initializeMemory();
  void *ptr = allocate(mem, 32);
  void *result = reallocate(mem, ptr, 0);
  ASSERT_NULL(result, "reallocate(ptr, 0) should return NULL");
  freeMemory(mem);
}

static void test_allocations_are_contiguous_at_end(void) {
  Memory *mem = initializeMemory();
  size_t size1 = 32;
  size_t size2 = 64;

  void *ptr1 = allocate(mem, size1);
  void *ptr2 = allocate(mem, size2);

  uint8_t *end_of_first_block = (uint8_t *)ptr1 + size1 + sizeof(blockfooter) + sizeof(blockheader);

  ASSERT_EQ((uint8_t *)ptr2, end_of_first_block,
            "second allocation should be right after first block");

  freeMemory(mem);
}

static void test_header_constructed_correctly(void) {
  Memory *mem = initializeMemory();
  size_t size = 48;

  void *ptr = allocate(mem, size);
  blockheader *header = BLOCK_HEADER(ptr);

  ASSERT_EQ(header->blockSize, size, "header blockSize should match requested size");
  ASSERT_EQ(header->free, false, "header free should be false after allocation");

  freeMemory(mem);
}

static void test_footer_constructed_correctly(void) {
  Memory *mem = initializeMemory();
  size_t size = 48;

  void *ptr = allocate(mem, size);
  uint8_t *arena = ARENA(mem);
  blockheader *header = BLOCK_HEADER(ptr);
  blockfooter *footer = (blockfooter *)((uint8_t *)ptr + size);

  size_t expected_offset = (uint8_t *)header - arena;
  ASSERT_EQ(footer->headerOffset, expected_offset, "footer headerOffset should point to header");

  blockheader *header_from_footer = (blockheader *)(arena + footer->headerOffset);
  ASSERT_EQ(header_from_footer, header, "footer headerOffset should resolve back to header");

  freeMemory(mem);
}

static void test_header_free_after_free(void) {
  Memory *mem = initializeMemory();
  void *ptr = allocate(mem, 32);
  blockheader *header = BLOCK_HEADER(ptr);

  ASSERT_EQ(header->free, false, "header free should be false before freeAlloc");
  freeAlloc(mem, ptr);
  ASSERT_EQ(header->free, true, "header free should be true after freeAlloc");

  freeMemory(mem);
}

static void test_coalesce_backwards_merges_blocks(void) {
  Memory *mem = initializeMemory();
  size_t size1 = 32;
  size_t size2 = 64;

  void *ptr1 = allocate(mem, size1);
  void *ptr2 = allocate(mem, size2);

  blockheader *header1 = BLOCK_HEADER(ptr1);
  blockheader *header2 = BLOCK_HEADER(ptr2);

  freeAlloc(mem, ptr1);
  ASSERT_EQ(header1->free, true, "first block should be free");
  ASSERT_EQ(header1->blockSize, size1, "first block size unchanged before merge");

  freeAlloc(mem, ptr2);

  size_t expected_merged_size = size1 + sizeof(blockfooter) + sizeof(blockheader) + size2;
  ASSERT_EQ(header1->blockSize, expected_merged_size,
            "first block should grow to include second block");
  ASSERT_EQ(header1->free, true, "merged block should be free");

  freeMemory(mem);
}

static void test_coalesce_backwards_updates_footer(void) {
  Memory *mem = initializeMemory();
  size_t size1 = 32;
  size_t size2 = 64;

  void *ptr1 = allocate(mem, size1);
  void *ptr2 = allocate(mem, size2);

  uint8_t *arena = ARENA(mem);
  blockheader *header1 = BLOCK_HEADER(ptr1);

  freeAlloc(mem, ptr1);
  freeAlloc(mem, ptr2);

  blockfooter *merged_footer = (blockfooter *)((uint8_t *)(header1 + 1) + header1->blockSize);
  size_t expected_offset = (uint8_t *)header1 - arena;

  ASSERT_EQ(merged_footer->headerOffset, expected_offset,
            "merged footer should point to first header");

  freeMemory(mem);
}

static void test_coalesce_backwards_not_when_prev_allocated(void) {
  Memory *mem = initializeMemory();
  size_t size1 = 32;
  size_t size2 = 64;

  void *ptr1 = allocate(mem, size1);
  void *ptr2 = allocate(mem, size2);

  blockheader *header1 = BLOCK_HEADER(ptr1);
  blockheader *header2 = BLOCK_HEADER(ptr2);

  freeAlloc(mem, ptr2);

  ASSERT_EQ(header1->free, false, "first block should still be allocated");
  ASSERT_EQ(header1->blockSize, size1, "first block size should be unchanged");
  ASSERT_EQ(header2->free, true, "second block should be free");
  ASSERT_EQ(header2->blockSize, size2, "second block size should be unchanged");

  freeMemory(mem);
}

static void test_find_free_space_from_beginning(void) {
  Memory *mem = initializeMemory();

  size_t block_size = 1024;
  size_t total_block = sizeof(blockheader) + block_size + sizeof(blockfooter);
  size_t num_blocks = mem->capacity / total_block;

  void **ptrs = malloc(num_blocks * sizeof(void *));
  for (size_t i = 0; i < num_blocks; i++) {
    ptrs[i] = allocate(mem, block_size);
  }

  ASSERT_NOT_NULL(ptrs[0], "first block should be allocated");
  ASSERT_NOT_NULL(ptrs[num_blocks - 1], "last block should be allocated");

  freeAlloc(mem, ptrs[0]);
  freeAlloc(mem, ptrs[2]);

  blockheader *header0 = BLOCK_HEADER(ptrs[0]);
  blockheader *header2 = BLOCK_HEADER(ptrs[2]);
  ASSERT_EQ(header0->free, true, "block 0 should be free");
  ASSERT_EQ(header2->free, true, "block 2 should be free");

  void *new_ptr = allocate(mem, block_size);
  ASSERT_NOT_NULL(new_ptr, "should find free space from beginning");
  ASSERT_EQ(new_ptr, ptrs[0], "should reuse first freed block");

  blockheader *reused_header = BLOCK_HEADER(new_ptr);
  ASSERT_EQ(reused_header->free, false, "reused block should be marked allocated");

  free(ptrs);
  freeMemory(mem);
}

static void test_find_free_space_skips_allocated(void) {
  Memory *mem = initializeMemory();

  size_t block_size = 1024;
  size_t total_block = sizeof(blockheader) + block_size + sizeof(blockfooter);
  size_t num_blocks = mem->capacity / total_block;

  void **ptrs = malloc(num_blocks * sizeof(void *));
  for (size_t i = 0; i < num_blocks; i++) {
    ptrs[i] = allocate(mem, block_size);
  }

  freeAlloc(mem, ptrs[2]);

  void *new_ptr = allocate(mem, block_size);
  ASSERT_EQ(new_ptr, ptrs[2], "should find and reuse block 2");

  free(ptrs);
  freeMemory(mem);
}

static void test_returns_null_when_out_of_memory(void) {
  Memory *mem = initializeMemory();

  size_t block_size = 1024;
  size_t total_block = sizeof(blockheader) + block_size + sizeof(blockfooter);
  size_t num_blocks = mem->capacity / total_block;

  void **ptrs = malloc(num_blocks * sizeof(void *));
  for (size_t i = 0; i < num_blocks; i++) {
    ptrs[i] = allocate(mem, block_size);
  }

  void *should_be_null = allocate(mem, block_size);
  ASSERT_NULL(should_be_null, "should return NULL when no space available");

  free(ptrs);
  freeMemory(mem);
}

void run_memory_tests(void) {
  printf("=== Memory Tests ===\n");
  test_allocate_returns_non_null();
  test_allocate_zero_returns_null();
  test_allocate_multiple_blocks();
  test_allocate_data_integrity();
  test_free_and_reuse();
  test_reallocate_grows_block();
  test_reallocate_null_allocates();
  test_reallocate_to_zero_frees();
  test_allocations_are_contiguous_at_end();
  test_header_constructed_correctly();
  test_footer_constructed_correctly();
  test_header_free_after_free();
  test_coalesce_backwards_merges_blocks();
  test_coalesce_backwards_updates_footer();
  test_coalesce_backwards_not_when_prev_allocated();
  test_find_free_space_from_beginning();
  test_find_free_space_skips_allocated();
  test_returns_null_when_out_of_memory();
}
