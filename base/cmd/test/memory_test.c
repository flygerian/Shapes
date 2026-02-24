#include "test.h"
#include "../../memory.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Total bytes consumed in the arena for a block of the given payload size.
static size_t totalBlockSize(size_t payloadSize) {
  return sizeof(blockheader) + payloadSize + sizeof(blockfooter);
}

// Fill the arena with enough same-sized blocks to exhaust capacity.
// Returns the number of blocks allocated and fills ptrs[].
static size_t fillArena(Memory *mem, size_t blockSize, void **ptrs, size_t maxPtrs) {
  size_t n = 0;
  while (n < maxPtrs) {
    void *p = allocate(mem, blockSize);
    if (p == NULL)
      break;
    ptrs[n++] = p;
  }
  return n;
}

// ---------------------------------------------------------------------------
// Allocation basics
// ---------------------------------------------------------------------------

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

static void test_allocate_marks_block_allocated(void) {
  Memory *mem = initializeMemory();
  void *ptr = allocate(mem, 48);
  blockheader *hdr = BLOCK_HEADER(ptr);
  ASSERT_EQ(hdr->free, false, "newly allocated block should have free=false");
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
  ASSERT_NEQ(ptr1, ptr2, "first and second allocations should be at different addresses");
  ASSERT_NEQ(ptr2, ptr3, "second and third allocations should be at different addresses");

  freeMemory(mem);
}

static void test_allocate_data_integrity(void) {
  Memory *mem = initializeMemory();
  char *str = allocate(mem, 16);
  strcpy(str, "hello");
  ASSERT_EQ(strcmp(str, "hello"), 0, "allocated memory should hold written data");
  freeMemory(mem);
}

// ---------------------------------------------------------------------------
// Block layout: header and footer construction
// ---------------------------------------------------------------------------

static void test_header_blocksize_matches_request(void) {
  Memory *mem = initializeMemory();
  size_t size = 48;
  void *ptr = allocate(mem, size);
  blockheader *hdr = BLOCK_HEADER(ptr);
  ASSERT_EQ(hdr->blockSize, size, "header blockSize should match requested size");
  freeMemory(mem);
}

static void test_footer_offset_resolves_to_header(void) {
  Memory *mem = initializeMemory();
  size_t size = 48;
  void *ptr = allocate(mem, size);
  uint8_t *arena = ARENA(mem);
  blockheader *hdr = BLOCK_HEADER(ptr);
  blockfooter *footer = (blockfooter *)((uint8_t *)ptr + size);

  size_t expectedOffset = (uint8_t *)hdr - arena;
  ASSERT_EQ(footer->headerOffset, expectedOffset,
            "footer headerOffset should equal header's arena offset");

  blockheader *hdrFromFooter = (blockheader *)(arena + footer->headerOffset);
  ASSERT_EQ(hdrFromFooter, hdr, "footer headerOffset should resolve back to the same header");

  freeMemory(mem);
}

static void test_allocations_are_contiguous_in_arena(void) {
  Memory *mem = initializeMemory();
  size_t size1 = 32;
  size_t size2 = 64;

  void *ptr1 = allocate(mem, size1);
  void *ptr2 = allocate(mem, size2);

  // ptr2 should start immediately after the footer of the first block.
  uint8_t *expectedPtr2 = (uint8_t *)ptr1 + size1 + sizeof(blockfooter) + sizeof(blockheader);
  ASSERT_EQ((uint8_t *)ptr2, expectedPtr2,
            "second allocation payload should start right after first block's footer + header");

  freeMemory(mem);
}

static void test_allocated_bytes_advances_by_total_block_size(void) {
  Memory *mem = initializeMemory();
  size_t sizeBefore = mem->allocated;
  size_t payloadSize = 128;
  allocate(mem, payloadSize);
  ASSERT_EQ(mem->allocated - sizeBefore, totalBlockSize(payloadSize),
            "allocated should advance by header + payload + footer");
  freeMemory(mem);
}

// ---------------------------------------------------------------------------
// Free and reuse
// ---------------------------------------------------------------------------

static void test_free_marks_block_free(void) {
  Memory *mem = initializeMemory();
  void *ptr = allocate(mem, 32);
  blockheader *hdr = BLOCK_HEADER(ptr);

  ASSERT_EQ(hdr->free, false, "block should be allocated before freeAlloc");
  freeAlloc(mem, ptr);
  ASSERT_EQ(hdr->free, true, "block should be marked free after freeAlloc");

  freeMemory(mem);
}

static void test_free_and_reuse(void) {
  Memory *mem = initializeMemory();
  void *ptr1 = allocate(mem, 64);
  freeAlloc(mem, ptr1);
  void *ptr2 = allocate(mem, 32);
  ASSERT_NOT_NULL(ptr2, "allocate should succeed after a free");
  freeMemory(mem);
}

// ---------------------------------------------------------------------------
// Block splitting: free remainder after partial reuse
// ---------------------------------------------------------------------------

// Fill the arena with one large block, then free it and re-allocate smaller
// to force findAvailableSpace to split the freed block.
static void test_split_creates_free_remainder(void) {
  size_t largeSize = 256;
  size_t smallSize = 64;
  Memory *mem = initializeArena(totalBlockSize(largeSize), 1);

  void *large = allocate(mem, largeSize);
  freeAlloc(mem, large);

  // Arena is now full (allocated == capacity), so the next allocate goes
  // through findAvailableSpace and splits the freed block.
  void *reused = allocate(mem, smallSize);
  ASSERT_NOT_NULL(reused, "should reuse the freed block for a smaller allocation");
  ASSERT_EQ(reused, large, "smaller allocation should reuse the start of the freed block");

  blockheader *reusedHdr = BLOCK_HEADER(reused);
  ASSERT_EQ(reusedHdr->blockSize, smallSize, "reused block header should reflect the smaller size");
  ASSERT_EQ(reusedHdr->free, false, "reused block should be marked allocated");

  // There should be a free remainder block immediately after.
  uint8_t *remainderStart = (uint8_t *)reused + smallSize + sizeof(blockfooter);
  blockheader *remainderHdr = (blockheader *)remainderStart;
  ASSERT_EQ(remainderHdr->free, true, "remainder block should be free after split");

  size_t expectedRemainderPayload = largeSize - smallSize - totalBlockSize(0);
  ASSERT_EQ(remainderHdr->blockSize, expectedRemainderPayload,
            "remainder block payload should be largeSize - smallSize - header/footer overhead");

  freeMemory(mem);
}

static void test_split_remainder_footer_offset_is_correct(void) {
  size_t largeSize = 256;
  size_t smallSize = 64;
  Memory *mem = initializeArena(totalBlockSize(largeSize), 1);
  uint8_t *arena = ARENA(mem);

  void *large = allocate(mem, largeSize);
  freeAlloc(mem, large);
  allocate(mem, smallSize);

  // Locate the remainder header.
  uint8_t *remainderStart = (uint8_t *)large + smallSize + sizeof(blockfooter);
  blockheader *remainderHdr = (blockheader *)remainderStart;

  blockfooter *remainderFooter =
      (blockfooter *)((uint8_t *)(remainderHdr + 1) + remainderHdr->blockSize);
  size_t expectedOffset = (uint8_t *)remainderHdr - arena;

  ASSERT_EQ(remainderFooter->headerOffset, expectedOffset,
            "remainder block footer should point back to remainder header");

  freeMemory(mem);
}

// ---------------------------------------------------------------------------
// Coalescing
// ---------------------------------------------------------------------------

static void test_coalesce_backwards_merges_adjacent_free_blocks(void) {
  Memory *mem = initializeMemory();
  size_t size1 = 32;
  size_t size2 = 64;

  void *ptr1 = allocate(mem, size1);
  void *ptr2 = allocate(mem, size2);

  blockheader *hdr1 = BLOCK_HEADER(ptr1);

  freeAlloc(mem, ptr1);
  ASSERT_EQ(hdr1->blockSize, size1, "first block size should be unchanged before merge");

  freeAlloc(mem, ptr2);

  size_t expectedMergedPayload = size1 + totalBlockSize(size2);
  ASSERT_EQ(hdr1->blockSize, expectedMergedPayload,
            "merged block payload should be size1 + full block size of size2");
  ASSERT_EQ(hdr1->free, true, "merged block should be marked free");

  freeMemory(mem);
}

static void test_coalesce_backwards_footer_points_to_merged_header(void) {
  Memory *mem = initializeMemory();
  size_t size1 = 32;
  size_t size2 = 64;
  uint8_t *arena = ARENA(mem);

  void *ptr1 = allocate(mem, size1);
  void *ptr2 = allocate(mem, size2);
  blockheader *hdr1 = BLOCK_HEADER(ptr1);

  freeAlloc(mem, ptr1);
  freeAlloc(mem, ptr2);

  blockfooter *mergedFooter = (blockfooter *)((uint8_t *)(hdr1 + 1) + hdr1->blockSize);
  size_t expectedOffset = (uint8_t *)hdr1 - arena;
  ASSERT_EQ(mergedFooter->headerOffset, expectedOffset,
            "merged block footer should point to the first (surviving) header");

  freeMemory(mem);
}

static void test_coalesce_does_not_merge_when_prev_allocated(void) {
  Memory *mem = initializeMemory();
  size_t size1 = 32;
  size_t size2 = 64;

  void *ptr1 = allocate(mem, size1);
  void *ptr2 = allocate(mem, size2);

  blockheader *hdr1 = BLOCK_HEADER(ptr1);
  blockheader *hdr2 = BLOCK_HEADER(ptr2);

  freeAlloc(mem, ptr2);

  ASSERT_EQ(hdr1->free, false, "first block should remain allocated");
  ASSERT_EQ(hdr1->blockSize, size1, "first block size should be unchanged");
  ASSERT_EQ(hdr2->free, true, "second block should be free");
  ASSERT_EQ(hdr2->blockSize, size2, "second block size should be unchanged (no merge)");

  freeMemory(mem);
}

static void test_coalesce_three_blocks_into_one(void) {
  Memory *mem = initializeMemory();
  size_t s1 = 32, s2 = 64, s3 = 128;

  void *p1 = allocate(mem, s1);
  void *p2 = allocate(mem, s2);
  void *p3 = allocate(mem, s3);
  blockheader *hdr1 = BLOCK_HEADER(p1);

  freeAlloc(mem, p1);
  freeAlloc(mem, p2);
  // After p1+p2 merged, freeing p3 should extend to one big block.
  freeAlloc(mem, p3);

  // Expected: s1 + full(s2) + full(s3)
  size_t expectedPayload = s1 + totalBlockSize(s2) + totalBlockSize(s3);
  ASSERT_EQ(hdr1->blockSize, expectedPayload,
            "three consecutive frees should coalesce into one block");
  ASSERT_EQ(hdr1->free, true, "fully coalesced block should be free");

  freeMemory(mem);
}

// ---------------------------------------------------------------------------
// Reallocation
// ---------------------------------------------------------------------------

static void test_reallocate_null_acts_as_allocate(void) {
  Memory *mem = initializeMemory();
  void *ptr = reallocate(mem, NULL, 32);
  ASSERT_NOT_NULL(ptr, "reallocate(NULL, size) should behave like allocate");
  freeMemory(mem);
}

static void test_reallocate_to_zero_frees_block(void) {
  Memory *mem = initializeMemory();
  void *ptr = allocate(mem, 32);
  void *result = reallocate(mem, ptr, 0);
  ASSERT_NULL(result, "reallocate(ptr, 0) should return NULL and free the block");
  blockheader *hdr = BLOCK_HEADER(ptr);
  ASSERT_EQ(hdr->free, true, "block should be marked free after reallocate(ptr, 0)");
  freeMemory(mem);
}

static void test_reallocate_smaller_returns_same_ptr(void) {
  Memory *mem = initializeMemory();
  void *ptr = allocate(mem, 64);
  void *result = reallocate(mem, ptr, 32);
  ASSERT_EQ(result, ptr, "reallocate to smaller size should return the same pointer");
  freeMemory(mem);
}

static void test_reallocate_grows_and_preserves_data(void) {
  Memory *mem = initializeMemory();
  char *ptr = allocate(mem, 16);
  strcpy(ptr, "test");
  ptr = reallocate(mem, ptr, 64);
  ASSERT_NOT_NULL(ptr, "reallocate to larger size should succeed");
  ASSERT_EQ(strcmp(ptr, "test"), 0, "reallocate should preserve original data");
  freeMemory(mem);
}

// ---------------------------------------------------------------------------
// Free-space scan and reuse
// ---------------------------------------------------------------------------

// Use a small fixed arena for fillArena tests to avoid O(n^2) scan behavior
// with the default 64MB arena.
#define SCAN_TEST_ARENA_SIZE (1024 * 64)

static void test_allocate_reuses_first_free_block(void) {
  Memory *mem = initializeArena(SCAN_TEST_ARENA_SIZE, 1);
  size_t blockSize = 1024;
  size_t capacity = mem->capacity / totalBlockSize(blockSize);

  void **ptrs = malloc(capacity * sizeof(void *));
  size_t n = fillArena(mem, blockSize, ptrs, capacity);

  ASSERT_NOT_NULL(ptrs[0], "first block should be allocated");
  ASSERT_NOT_NULL(ptrs[n - 1], "last block should be allocated");

  freeAlloc(mem, ptrs[0]);
  freeAlloc(mem, ptrs[2]);

  void *newPtr = allocate(mem, blockSize);
  ASSERT_NOT_NULL(newPtr, "should find free space");
  ASSERT_EQ(newPtr, ptrs[0], "should reuse the first freed block (block 0)");

  blockheader *reusedHdr = BLOCK_HEADER(newPtr);
  ASSERT_EQ(reusedHdr->free, false, "reused block should be marked allocated");

  free(ptrs);
  freeMemory(mem);
}

static void test_allocate_skips_allocated_blocks(void) {
  Memory *mem = initializeArena(SCAN_TEST_ARENA_SIZE, 1);
  size_t blockSize = 1024;
  size_t capacity = mem->capacity / totalBlockSize(blockSize);

  void **ptrs = malloc(capacity * sizeof(void *));
  fillArena(mem, blockSize, ptrs, capacity);

  freeAlloc(mem, ptrs[2]);

  void *newPtr = allocate(mem, blockSize);
  ASSERT_EQ(newPtr, ptrs[2], "should find and reuse block 2");

  free(ptrs);
  freeMemory(mem);
}

static void test_returns_null_when_out_of_memory(void) {
  Memory *mem = initializeArena(SCAN_TEST_ARENA_SIZE, 1);
  size_t blockSize = 1024;
  size_t capacity = mem->capacity / totalBlockSize(blockSize);

  void **ptrs = malloc(capacity * sizeof(void *));
  fillArena(mem, blockSize, ptrs, capacity);

  void *shouldBeNull = allocate(mem, blockSize);
  ASSERT_NULL(shouldBeNull, "allocate should return NULL when arena is full");

  free(ptrs);
  freeMemory(mem);
}

// ---------------------------------------------------------------------------
// numFreeBlocks tracking
// ---------------------------------------------------------------------------

static void test_num_free_blocks_starts_at_zero(void) {
  Memory *mem = initializeMemory();
  ASSERT_EQ(mem->numFreeBlocks, (size_t)0, "numFreeBlocks should be 0 on init");
  freeMemory(mem);
}

static void test_num_free_blocks_unchanged_after_allocate(void) {
  Memory *mem = initializeMemory();
  allocate(mem, 64);
  ASSERT_EQ(mem->numFreeBlocks, (size_t)0, "numFreeBlocks should stay 0 after allocate");
  freeMemory(mem);
}

static void test_num_free_blocks_increments_on_free(void) {
  Memory *mem = initializeMemory();
  void *ptr = allocate(mem, 64);
  freeAlloc(mem, ptr);
  ASSERT_EQ(mem->numFreeBlocks, (size_t)1, "numFreeBlocks should be 1 after freeing one block");
  freeMemory(mem);
}

static void test_num_free_blocks_decrements_on_reuse(void) {
  size_t blockSz = 256;
  Memory *mem = initializeArena(totalBlockSize(blockSz), 1);

  void *ptr = allocate(mem, blockSz);
  freeAlloc(mem, ptr);
  ASSERT_EQ(mem->numFreeBlocks, (size_t)1, "numFreeBlocks should be 1 after free");

  // Arena is full so next allocate goes through findAvailableSpace; no split
  // possible because the leftover would be zero-sized.
  allocate(mem, blockSz);
  ASSERT_EQ(mem->numFreeBlocks, (size_t)0,
            "numFreeBlocks should be 0 after reusing the free block");
  freeMemory(mem);
}

static void test_num_free_blocks_split_adds_free_remainder(void) {
  size_t largeSize = 256;
  size_t smallSize = 64;
  Memory *mem = initializeArena(totalBlockSize(largeSize), 1);

  void *large = allocate(mem, largeSize);
  freeAlloc(mem, large);
  ASSERT_EQ(mem->numFreeBlocks, (size_t)1, "numFreeBlocks should be 1 after freeing large block");

  // Allocating smaller triggers a split: the reused block becomes allocated and
  // the remainder is a new free block, so numFreeBlocks stays at 1.
  allocate(mem, smallSize);
  ASSERT_EQ(mem->numFreeBlocks, (size_t)1,
            "numFreeBlocks should be 1 after split (reused block consumed, remainder free)");
  freeMemory(mem);
}

static void test_num_free_blocks_coalesce_reduces_count(void) {
  Memory *mem = initializeMemory();
  void *ptr1 = allocate(mem, 32);
  void *ptr2 = allocate(mem, 64);

  freeAlloc(mem, ptr1);
  ASSERT_EQ(mem->numFreeBlocks, (size_t)1, "numFreeBlocks should be 1 after first free");

  // Freeing the adjacent block triggers coalesceBackwards, merging two free
  // blocks into one, so numFreeBlocks goes back to 1 (not 2).
  freeAlloc(mem, ptr2);
  ASSERT_EQ(mem->numFreeBlocks, (size_t)1,
            "numFreeBlocks should stay 1 after coalescing two adjacent free blocks");
  freeMemory(mem);
}

static void test_num_free_blocks_multiple_isolated_free_blocks(void) {
  Memory *mem = initializeMemory();
  void *ptr1 = allocate(mem, 32);
  allocate(mem, 16); // separator — keeps ptr1 and ptr3 non-adjacent
  void *ptr3 = allocate(mem, 64);
  allocate(mem, 16); // separator

  freeAlloc(mem, ptr1);
  freeAlloc(mem, ptr3);

  ASSERT_EQ(mem->numFreeBlocks, (size_t)2,
            "numFreeBlocks should be 2 for two non-adjacent free blocks");
  freeMemory(mem);
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------

void run_memory_tests(void) {
  printf("=== Memory Tests ===\n");

  // Allocation basics
  test_allocate_returns_non_null();
  test_allocate_zero_returns_null();
  test_allocate_marks_block_allocated();
  test_allocate_multiple_blocks();
  test_allocate_data_integrity();

  // Block layout
  test_header_blocksize_matches_request();
  test_footer_offset_resolves_to_header();
  test_allocations_are_contiguous_in_arena();
  test_allocated_bytes_advances_by_total_block_size();

  // Free and reuse
  test_free_marks_block_free();
  test_free_and_reuse();

  // Splitting
  test_split_creates_free_remainder();
  test_split_remainder_footer_offset_is_correct();

  // Coalescing
  test_coalesce_backwards_merges_adjacent_free_blocks();
  test_coalesce_backwards_footer_points_to_merged_header();
  test_coalesce_does_not_merge_when_prev_allocated();
  test_coalesce_three_blocks_into_one();

  // Reallocation
  test_reallocate_null_acts_as_allocate();
  test_reallocate_to_zero_frees_block();
  test_reallocate_smaller_returns_same_ptr();
  test_reallocate_grows_and_preserves_data();

  // Free-space scan
  test_allocate_reuses_first_free_block();
  test_allocate_skips_allocated_blocks();
  test_returns_null_when_out_of_memory();

  // numFreeBlocks tracking
  test_num_free_blocks_starts_at_zero();
  test_num_free_blocks_unchanged_after_allocate();
  test_num_free_blocks_increments_on_free();
  test_num_free_blocks_decrements_on_reuse();
  test_num_free_blocks_split_adds_free_remainder();
  test_num_free_blocks_coalesce_reduces_count();
  test_num_free_blocks_multiple_isolated_free_blocks();
}
