1) Redesign Context ownership and tracking (go/context.go)
Add root ownership
- Add a root *Context pointer on Context.
  - For a root context created via New(...), set root = self.
  - For derived contexts (NoGrad, NoGraph, Fused, and new “op subcontexts”), set root to the original root.
- Add ownsMemory bool (or equivalent) so only the root frees the C arena in Close().
  - Derived contexts’ Close() must be a no-op (or at least not free memory).
Split tracking lists
- Replace/augment existing fields with:
  - handles []*unsafe.Pointer on the root context only (used to nil tensor pointers on close).
  - locals []*unsafe.Pointer on every context (tensors created through that ctx).
  - intermediates []unsafe.Pointer on the root context only (C tensor pointers queued for FreeIntermediates).
Update Track semantics
- Change (*Context).Track(p *unsafe.Pointer) so it:
  - Always appends p to c.root.handles (so root Close() can nil every tensor pointer, regardless of which subctx created it).
  - Also appends p to c.locals (so this particular ctx can later infer intermediates).
Update MarkIntermediate semantics
- Change (*Context).MarkIntermediate(ptr unsafe.Pointer) so it always appends into c.root.intermediates (never the derived ctx’s own slice).
- Intermediates() and ClearIntermediates() should operate on the root’s list (or forward to root).
Update derived contexts
- NoGrad(), NoGraph(), Fused() should:
  - Share the same cCtx (arena) and root
  - Have fresh locals slices (do NOT share locals with parent; locals are “created by this ctx”)
  - Never copy intermediates/handles slices; those are effectively root-owned now
- Keep GradEnabled / BackwardEnabled behavior consistent with today.
Add a generic subcontext constructor
- Add an internal helper (e.g., derive(...)) to avoid inconsistent setup between NoGrad/NoGraph/Fused.
Acceptance checks
- Verify that if you allocate tensors via a Fused() ctx, the root Close() still nils their Go pointers.
---
2) Add “Saved tensors” to the computation graph node (go/grad.go)
Change ComputationGraphNode
- Add a field: Saved []*Tensor
  - Contract: “tensors required later in backward, created during forward, but not necessarily mathematical inputs.”
Add a new API for node creation with saved tensors
- Keep the existing method:
  - ctx.NewComputationGraphNode(result, op, backward, inputs...)
- Add a new method:
  - ctx.NewComputationGraphNodeSaved(result, op, backward, saved []*Tensor, inputs ...*Tensor)
- Internally, both should call a shared helper that:
  - Attaches the node (same as today)
  - Records Saved on the node
  - If ctx is a fused/subcontext, triggers the “mark forward intermediates” sweep (next step)
Why this is needed
- Today CrossEntropy stores probs in node.Metadata after node creation. That’s incompatible with “sweep locals at node creation time” because the sweeper can’t see that probs must be kept.
- Saved makes “keep across forward/backward” explicit and sweep-safe.
---
3) Implement local sweeping to mark intermediates (forward fused + backward op ctx)
Create a helper in the root shapes package (likely go/grad.go or a small new file):
- markIntermediatesFromLocals(ctx *Context, keepTensors ...*Tensor)
  - Build a keep-set keyed by the C tensor pointer (unsafe.Pointer(t.cTensor))
  - Iterate ctx.locals:
    - Each entry is *unsafe.Pointer handle; dereference it to get the C tensor pointer
    - If nil, skip (already invalidated)
    - If not in keep-set: call ctx.MarkIntermediate(ctPtr) (which routes to root)
  - Clear ctx.locals after sweeping
Important: this function only marks. It must not free.
Dedup safety
- Update FreeIntermediates (next step) to deduplicate pointers before freeing, to prevent double-free if something is marked twice.
---
4) Update fused behavior: sweep locals when creating the custom node
Where: go/grad.go, inside Context.NewComputationGraphNode*
Rule
- When a computation is performed in a fused context, you call fusedCtx.NewComputationGraphNode...(...) (not the parent ctx).
- Immediately after attaching the node, call the sweeper:
  - Keep:
    - result
    - all inputs
    - all saved (if using the Saved API)
  - Mark everything else created under that fused ctx as intermediate
Result
- Layers and fused composite ops no longer need to manually call MarkIntermediate for forward temporaries.
---
5) Update backward behavior: per-node op subcontext + sweep locals
Where: (*Tensor).Backward in go/grad.go
Change the backward loop
- Instead of calling node.Backward(noGraphCtx, node) directly:
  - Create a fresh op context per node: opCtx := noGraphCtx.<derive-op-subcontext>()
    - Shares same arena + same root
    - BackwardEnabled=false, GradEnabled=false
    - Fresh locals
  - Call node.Backward(opCtx, node)
  - After backward returns, sweep opCtx.locals:
    - Keep: for each input tensor inp := range node.Inputs, keep inp.Computation.Grad
      - (Because many backward funcs allocate a new grad tensor via .Plus(...) and assign it.)
    - Also keep node.Grad if it can be created under opCtx (typically it won’t be, but safe to include)
  - Mark node.Saved as intermediate after the node runs (because saved tensors are only needed up to that node’s backward execution).
End of backward
- Keep the current behavior: FreeIntermediates(noGraphCtx) at the end.
  - Ensure FreeIntermediates reads from the root’s intermediates list.
Why this works
- Any temporary tensor created inside an op’s backward function will be tracked in opCtx.locals and automatically marked.
- No more markIntermediate calls in op backward implementations.
---
6) Make FreeIntermediates root-based and deduped (go/grad.go)
Root-based
- FreeIntermediates(ctx) should operate on ctx.root.intermediates.
Deduplicate
- Before freeing, build a map[unsafe.Pointer]bool to ensure each C tensor pointer is freed once.
Free logic
- Keep existing logic:
  - Cast to *C.Tensor
  - Use ct.isView to choose FreeViewTensor vs FreeTensor
---
7) Update call sites to actually use fused contexts consistently
This is critical; otherwise forward temporaries won’t be created in the fused ctx and won’t be swept.
go/layer/dense.go
- Ensure all internal forward ops (including UnSqueeze, Squeeze, Transpose, etc.) run under fusedCtx, not the parent ctx.
- Change node registration to call on fusedCtx:
  - fusedCtx.NewComputationGraphNode(...) (or Saved variant if needed)
go/activation/activation.go (Tanh)
- Replace the current “build forward under NoGrad()” approach with fused forward:
  - Compute internal temps using fusedCtx := ctx.Fused()
  - Register custom node via fusedCtx.NewComputationGraphNode(...)
  - This allows forward temporaries to be swept automatically.
go/loss/cross_entropy.go
- Switch from node.Metadata = probs to using the Saved API:
  - Use NewComputationGraphNodeSaved(result, OpCrossEntropy, backward, saved=[probs], inputs=[yGround, logits])
- Keep non-tensor metadata (dims, ranges) as Metadata where appropriate.
Check other Metadata assignments
- go/access.go stores index tensors in Metadata. That’s fine as long as those tensors are not fused temporaries that should be swept; if they are, they should be in Inputs or Saved. Document this rule for future ops.
---
8) Remove manual intermediate marking across the codebase (after sweeping works)
Once steps 1–7 pass tests, delete the now-redundant calls:
- In root package backward implementations:
  - go/grad_binary.go
  - go/grad_shape.go
  - go/grad_access.go
- In subpackages:
  - go/loss/mse.go (forward manual marking)
  - go/loss/cross_entropy.go (backward manual marking becomes unnecessary once opCtx sweeping works)
- Keep exported helpers MarkIntermediate/MarkIfIntermediate for compatibility, but they should be rarely needed.
Note about MarkIfIntermediate
- Some current code uses MarkIfIntermediate to avoid freeing an origin tensor when a function returns the input unchanged.
- With opCtx sweeping, this becomes irrelevant because “origin tensors” are not in opCtx.locals (only newly created tensors are). So you can remove those patterns safely once the new tracking is correct.
---
9) Testing / validation checklist
Run Go tests after each major milestone:
1) After Context refactor (root/locals/handles): make test-go
2) After introducing Saved + updating CrossEntropy: make test-go
3) After backward opCtx sweeping: make test-go
4) After removing manual marks: make test-go
Specific tests to pay attention to:
- go/loss/cross_entropy_test.go (saved tensor correctness)
- go/layer/dense_test.go (fused internals + custom node backward)
- go/grad_test.go (basic ops, chain rule)
- Any indexing backward tests (go/grad_access_test.go) to ensure metadata tensors remain valid
Add one new test (recommended):
- Create a fused composite op that saves a forward tensor and ensure it survives until backward, then is freed (can be indirect: run backward twice or allocate many iterations and ensure no blow-up / no crash).
---
Operational Rules to Document (for future contributors)
- If a custom backward needs a forward-created tensor:
  - Put it in node.Saved (preferred) or as an explicit Input.
  - Do NOT hide it only in Metadata.
- Any composite/fused forward should:
  - Run internal ops using the fused ctx
  - Register the node using the fused ctx (so the sweep runs)
