Indexing + Slice Backward Plan (Views, VM-Friendly)


Goal recap
- Fix a C bug in view offset/boundary handling.
- Enable backward passes for:
  - coordinate indexing (`GetTensorAt` chain / `Tensor.Get(i, j, ...)`) as a view op
  - `Slice` as a view op
  - advanced indexing (`IndexWithTensor`, `IndexWithTensor2d`) via a single accumulate primitive
- Make view ops safe under intermediate freeing (no shared `boundary` pointers; correct transforms under transpose/squeeze/unsqueeze).
- Keep VM surface clean: view is a tensor property produced by ops; view-index backward uses grad-view + `AddInPlace`.
-------------------------------------------------------------------------------
 A. Core invariants (must hold)
 A1) `Tensor.boundary` meaning (C)
- `boundary == NULL` means no per-dimension offsets.
- Otherwise `boundary` points to an array of `Range` with length exactly `shape.numOfDims`.
- Logical coordinate `coord[d]` maps to underlying coordinate `(coord[d] + boundary[d].start)`.
 A2) Storage index computation
- Coordinate -> storage index computation must go only through:
  - `getContigousIdxFromCoord(Tensor*, dim_t*)`
- Callers must not add boundary offsets again.
 A3) `GetTensorAt` view representation
- `GetTensorAt` returns a view.
- It must not use a special "flat boundary" representation.
- The selected dim0 offset must be represented by shifting `dest.values` pointer, keeping `boundary` per-dim (or NULL).
 A4) Safe view freeing
- Any view op that returns a view must allocate its own `boundary` array when `source->boundary != NULL`.
- Boundary must be transformed correctly for transpose/squeeze/unsqueeze.
- Reason: `FreeViewTensor()` frees `t->boundary`, so shared pointers cause double-free/UAF when intermediates are freed after backward.
-------------------------------------------------------------------------------
 B. Phase 1 — Add failing C tests that capture current bugs
File: `base/cmd/test/tensor_test.c`
Add tests:
1) Slice boundary propagation correctness
- Base tensor `x` with known values.
- `s1 = Slice(x, ranges...)` with non-zero start in multiple dims.
- `GetAt(&s1, ...)` should match expected base values.
2) Nested slice correctness
- `s1 = Slice(x, ...)`
- `s2 = Slice(&s1, ...)`
- Validate `GetAt(&s2, ...)` matches base.
3) `GetTensorAt` on a sliced tensor
- `s = Slice(x, ...)`
- `row = GetTensorAt(s, k)`
- Validate `GetAt(&row, ...)` matches expected.
4) Advanced indexing on view input (read correctness)
- `s = Slice(x, ...)`
- `y = IndexWithTensor(s, indices)` and/or `IndexWithTensor2d(s, rowIdx, colIdx)`
- Assert values match expected.
Verification: run C tests after adding these.
-------------------------------------------------------------------------------
 C. Phase 2 — Fix view offset/boundary logic in C
 C1) Fix `Slice` boundary propagation
File: `base/tensor/shape_ops.c`
Bug: uses `source->boundary->start` instead of per-dim.
Fix: when `source->isView`:
- `boundary[x].start = source->boundary[x].start + ranges[x].start`
- `boundary[x].end   = source->boundary[x].start + ranges[x].end`
 C2) Fix `GetAt` double offset
File: `base/tensor/access.c`
Bug: `GetAt` adds `t->boundary->start` after `getContigousIdxFromCoord`.
Fix: remove the post-add entirely. `GetAt` uses only `getContigousIdxFromCoord`.
 C3) Fix `GetScalar` to not depend on boundary
File: `base/tensor/access.c`
Fix: always read the single element from `t->values` at index 0. No boundary logic.
 C4) Rewrite `GetTensorAt` view representation
File: `base/tensor/access.c`
Stop using "flat boundary". Always treat boundary as per-dimension.
Let:
- `bytesPerElem = getBytesForDtype(source->dtype)`
- `baseDim0Start = (source->isView && source->boundary) ? source->boundary[0].start : 0`
- `selected = baseDim0Start + index`
- `offsetElems = selected * source->shape.multipliers[0]`
- `dest.values = (char*)source->values + offsetElems * bytesPerElem`
Case newNumDims == 0 (scalar):
- `dest.shape.numOfDims = 0`
- `dest.shape.dims = NULL`
- `dest.shape.multipliers = NULL`
- `dest.boundary = NULL`
- `dest.size = 1`
- `dest.isView = true`
Case newNumDims > 0:
- allocate `dest.shape.dims[newNumDims]` and copy `source.shape.dims[1:]`
- allocate `dest.shape.multipliers[newNumDims]` and copy `source.shape.multipliers[1:]`
  - do not recompute multipliers; preserve stride for transposed views
- boundary:
  - if `source->boundary != NULL`, allocate `dest.boundary[newNumDims]` and copy `source.boundary[1:]`
  - else `dest.boundary = NULL`
- `dest.isView = true`
- `dest.isContigous = false`
- `dest.size = source.size / source.shape.dims[0]`
 C5) Fix advanced indexing offset logic for view inputs
File: `base/tensor/access.c`
Remove `source->boundary->start` special-casing.
Compute source slice base offsets using `getContigousIdxFromCoord`:
- 1D indexing: coords `[idx, 0, 0, ...]`
- 2D indexing: coords `[rowIdx, colIdx, 0, 0, ...]`
Memcpy from `source->values + srcOffset*bytesPerElem`.
Verification after Phase 2: run C tests; Phase 1 tests should now pass.
-------------------------------------------------------------------------------
 D. Phase 3 — Boundary ownership + transforms for ALL view-producing shape ops (follow-up)
File: `base/tensor/shape_ops.c`
Problem: view ops often do `dest.boundary = source.boundary` (shared pointer). Also `Transpose` must permute boundary.
Fix requirements (A4):
1) `Transpose`
- if `source->boundary == NULL`: `dest.boundary = NULL`
- else allocate `newBoundary[ndims]`, copy, then swap entries `[d0]` and `[d1]`
2) `Squeeze`
- if `source->boundary == NULL`: `dest.boundary = NULL`
- else allocate `newBoundary[newNumDims]` and copy boundaries for surviving dims in the same order as `newDims`
3) `SqueezeDim`
- same, but remove the specified dim
4) `UnSqueeze`
- if `source->boundary == NULL`: `dest.boundary = NULL`
- else allocate `newBoundary[ndims+1]`
  - inserted dim boundary: `{start:0, end:1}`
  - copy old boundaries into other slots, shifted
Add C tests to catch sharing/double-free:
- start from a boundary-owning view (Slice)
- apply transpose/squeeze/unsqueeze to make another view
- `FreeViewTensor` on one view and verify the other still works (`GetAt` correct)
Do this before enabling Go backward intermediate freeing.
-------------------------------------------------------------------------------
 E. Phase 4 — Make `AddInPlace` support views (enabler for view backward)
File: `base/tensor/binary_op.c`
Goal: allow `a` to be view/non-contiguous. Must write underlying storage with boundary + multipliers.
Implementation outline:
- validate dtype + broadcast as before
- handle padding/contiguous conversion for `b` as today
- loop `x := 0..a->size-1`:
  - `unravel_index(x, &a->shape, aCoords)`
  - `aStorageIdx = getContigousIdxFromCoord(a, aCoords)`
  - compute `bCoords` via broadcast mapping
  - `bStorageIdx = getContigousIdxFromCoord(opB, bCoords)`
  - read aVal, bVal; write aVal+bVal into `a->values[aStorageIdx]`
Add C tests:
- `AddInPlace(sliceView, ones)` mutates correct region of base
- `AddInPlace(GetTensorAt(view), dy)` mutates correct slice in base
-------------------------------------------------------------------------------
 F. Phase 5 — Fix Go/CGo allocation for indexing ops (required)
File: `go/access.go`
Current bug: `getWithCoords` uses `var result C.Tensor` and stores `&result` in Go -> invalid to free.
Fix:
- Change CGo wrappers to allocate `Tensor*` on the arena and return via `Tensor** out` (like other ops).
  - `wrap_GetTensorAt(ctx, src, index, Tensor **out)`
  - `wrap_IndexWithTensor(ctx, src, indices, Tensor **out)`
  - `wrap_IndexWithTensor2d(ctx, src, row, col, Tensor **out)`
Go code uses `destPtr *C.Tensor`, wraps with `track(ctx, &Tensor{cTensor: destPtr})`.
-------------------------------------------------------------------------------
 G. Phase 6 — Autograd wiring for view-based indexing (GetTensorAt chain + Slice)
 G1) Add OpTypes
File: `go/grad.go`
- Add `OpGetTensorAt`, `OpSlice`, `OpIndexWithTensor`, `OpIndexWithTensor2d`
- Add to `String()`.
 G2) Attach nodes
File: `go/access.go`
- After each `GetTensorAt` step:
  - `attachNode(ctx, out, OpGetTensorAt, getTensorAtBackward, currentInput)`
  - `out.Computation.Metadata = idx (uint32)`
File: `go/shape_ops.go`
- In `Slice`, attach node with deep-copied `ranges` stored in metadata.
Attachment gating:
- Use `ctx.BackwardEnabled` for attaching nodes (important for `ctx.Fused()`).
 G3) Backward fns
New file: `go/grad_access.go`
`getTensorAtBackward`:
- `x := node.Inputs[0]`, `idx := node.Metadata.(uint32)`
- `dxView := x.Grad().Get(ctx, idx)`
- `dxView.AddInPlace(ctx, node.Grad)`
- mark `dxView` intermediate (view tensor) so it is freed after backward
`sliceBackward`:
- `ranges := node.Metadata.([]Range)`
- `dxView := x.Grad().Slice(ctx, ranges...)`
- `dxView.AddInPlace(ctx, node.Grad)`
- mark intermediate
Go tests (add):
- grad through `Get` (row selection) lands only on that row
- grad through `Slice` lands only in slice region
- nested views: `Slice(...).Get(...)` backward
-------------------------------------------------------------------------------
 H. Phase 7 — Advanced indexing backward (single primitive; VM-friendly segments optional)
You cannot avoid an accumulate primitive without very slow per-index loops.
 H1) Implement C accumulate kernels
File: `base/tensor/access.c` (or new file)
- `IndexAccumulate1d(ctx, dest, indices, srcGrad)`
- `IndexAccumulate2d(ctx, dest, row, col, srcGrad)`
These do in-place accumulation and support repeated indices.
 H2) Optional: VM segments "mini-opcodes" executor
Record layout: 8 u64 slots per record:
- slot0: destOffsetElems (u64)
- slot1: srcOffsetElems (u64)
- slot2: countElems (validate fits u32)
- slot3..7 reserved
Kernel:
- `AccumulateSegments(ctx, dest, src, u64 *records, u64 nRecords)`
Offsets use u64; count is u32 (stored as u64 + validated).
 H3) Go backward wiring
File: `go/access.go` (forward)
- attach nodes for `getWithTensor` and `getWithTensor2d`
File: `go/grad_access.go` (backward)
- call `IndexAccumulate1d/2d` on `x.Grad()` with indices and `node.Grad`
Go tests:
- 1D advanced indexing with repeats accumulates correctly
- 2D advanced indexing with repeats accumulates correctly
- advanced indexing on a view input + slice backward correctness
-------------------------------------------------------------------------------
 I. Phase 8 — Make Go node-attachment gating consistent
Policy:
- leaf grad buffers: gate on `GradEnabled`
- attach backward nodes: gate on `BackwardEnabled`
Update shape ops that currently use `GradEnabled` for node attachment to use `BackwardEnabled` where appropriate.
Add/keep tests around `ctx.Fused()` behavior.
-------------------------------------------------------------------------------
 J. Final verification
C:
- `make test` (or `make test-direct`)
Go:
- `make test-go`
- and/or:
  - `cd go && LD_LIBRARY_PATH=$PWD/../base/build/openblas/lib go test -v ./...`
Sanity:
- slice -> index -> sum -> backward should not panic; grads correct; no crash.
-------------------------------------------------------------------------------
 K. Common pitfalls
- Ensure `rg "boundary->start"` returns 0 after Phase 2+.
- Ensure `GetTensorAt` never allocates a single `Range` as a flat offset boundary.
- Ensure `GetScalar` reads from `values[0]` after scalar views shift `values`.
- Ensure view ops deep-copy boundary arrays; otherwise freeing intermediates will crash.
- Fix `go/access.go` wrappers to allocate `Tensor*` in C arena (no `var C.Tensor` + `&`).
