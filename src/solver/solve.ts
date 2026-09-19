// Solver: lock a few cues at fixed starts and remove all overlaps.
//
// Variables x[i] >= 0 are integer cue starts; feasibility constraints are
//   x[0] >= 0
//   x[i+1] >= x[i] + duration[i]
//   x[n-1] <= DAY_MS
// Pins force x[i] = start exactly. Objective: minimise sum |x[i] - base[i]|,
// ties broken by the lexicographically smallest vector (cue order).
//
// Change of variable y[i] = x[i] - P[i], where P[i] = sum_{k<i} duration[k].
// Then the gap constraints become y non-decreasing and the objective is
// sum |y[i] - b[i]| with b[i] = base[i] - P[i] — L1 isotonic regression with
// pinned observations. The lexicographically smallest minimiser is produced by
// PAVA blocks taking their *lower weighted median*; pins are observations with
// weight > total finite weight, so every block containing a pin evaluates to
// the pin value. Pins split the problem into independent segments, each solved
// by a PAVA stack. Each block keeps its observations in two leftist heaps
// (lower max-heap / upper min-heap) so its weighted lower median is the lower
// heap's top, restored after each merge in O(log n) heap operations.

export const DAY_MS = 86_400_000;

export interface Cue {
  start: number;
  duration: number;
  text: string;
}

/** cueIndex -> fixed integer start; one pin per cue, re-edit overwrites. */
export type Pins = ReadonlyMap<number, number>;

export interface SolveInput {
  cues: ReadonlyArray<Cue>;
  base: ReadonlyArray<number>;
  pins?: Pins;
}

export type SolveResult =
  | { ok: true; starts: number[]; cost: number }
  | { ok: false; reason: 'INFEASIBLE' };

/** Same contract as solve() with a configurable day span (tests use small U). */
export function solveWithSpan(
  cues: ReadonlyArray<Cue>,
  base: ReadonlyArray<number>,
  pins: Pins,
  daySpan: number,
): SolveResult {
  return solveCore(cues, base, pins, daySpan);
}

// ---------------------------------------------------------------------------
// PAVA blocks.
//
// A block keeps its observations in two leftist heaps so the weighted lower
// median is always the top of the lower heap:
//   lower: max-heap by value, holding the lighter-valued half,
//   upper: min-heap by value, holding the rest,
// with every lower value <= every upper value and the weight invariants
//   2 * lowerWeight >= total,
//   2 * (lowerWeight - top(lower).weight) < total,
// which pin the median at top(lower).value exactly. Merging two blocks merges
// the heaps pairwise (O(log n) each), then restores the invariants by swapping
// misordered tops and rebalancing weights — no per-merge array copies, so a
// full PAVA run is O(n log n) heap operations instead of O(n^2) merges.
// ---------------------------------------------------------------------------

interface HeapNode {
  value: number;
  weight: number;
  npl: number; // null-path length of the subtree rooted here
  left: HeapNode | null;
  right: HeapNode | null;
}

interface Block {
  lower: HeapNode | null; // max-heap by value
  upper: HeapNode | null; // min-heap by value
  lowerWeight: number; // total observation weight held in `lower`
  total: number;
  median: number; // always the lower-heap top value
  memberHead: EntryNode | null; // linked list of observations, index ascending
  memberTail: EntryNode | null;
}

interface EntryNode {
  index: number;
  next: EntryNode | null;
}

function rankOf(node: HeapNode | null): number {
  return node === null ? 0 : node.npl;
}

function heapNode(value: number, weight: number): HeapNode {
  return { value, weight, npl: 1, left: null, right: null };
}

/** Detach a popped root so it can re-enter a heap as a fresh singleton. */
function resetNode(node: HeapNode): HeapNode {
  node.left = null;
  node.right = null;
  node.npl = 1;
  return node;
}

/** Leftist heap merge; `keepsTop(a, b)` decides which root stays on top. */
function mergeHeaps(
  a: HeapNode | null,
  b: HeapNode | null,
  keepsTop: (x: HeapNode, y: HeapNode) => boolean,
): HeapNode | null {
  if (a === null) return b;
  if (b === null) return a;
  let top = a;
  let demoted = b;
  if (!keepsTop(a, b)) {
    top = b;
    demoted = a;
  }
  top.right = mergeHeaps(top.right, demoted, keepsTop);
  if (rankOf(top.left) < rankOf(top.right)) {
    const swap = top.left;
    top.left = top.right;
    top.right = swap;
  }
  top.npl = rankOf(top.right) + 1;
  return top;
}

const lowerFirst = (x: HeapNode, y: HeapNode): boolean => x.value >= y.value;
const upperFirst = (x: HeapNode, y: HeapNode): boolean => x.value <= y.value;

const mergeLower = (a: HeapNode | null, b: HeapNode | null): HeapNode | null =>
  mergeHeaps(a, b, lowerFirst);
const mergeUpper = (a: HeapNode | null, b: HeapNode | null): HeapNode | null =>
  mergeHeaps(a, b, upperFirst);

function singleton(index: number, value: number, weight: number): Block {
  const head: EntryNode = { index, next: null };
  return {
    lower: heapNode(value, weight),
    upper: null,
    lowerWeight: weight,
    total: weight,
    median: value,
    memberHead: head,
    memberTail: head,
  };
}

/** Merge PAVA block b into a (a precedes b); the median stays on the lower top. */
function mergeBlocks(a: Block, b: Block): Block {
  let lower = mergeLower(a.lower, b.lower);
  let upper = mergeUpper(a.upper, b.upper);
  let lowerWeight = a.lowerWeight + b.lowerWeight;
  const total = a.total + b.total;

  // Restore value order: every lower value must be <= every upper value.
  while (lower !== null && upper !== null && lower.value > upper.value) {
    const loTop = lower;
    lower = mergeLower(lower.left, lower.right);
    const upTop = upper;
    upper = mergeUpper(upper.left, upper.right);
    lowerWeight += upTop.weight - loTop.weight;
    lower = mergeLower(lower, resetNode(upTop));
    upper = mergeUpper(upper, resetNode(loTop));
  }

  // Rebalance weights so the weighted lower median sits on the lower top:
  //   2 * lowerWeight >= total and 2 * (lowerWeight - top.weight) < total.
  while (lower !== null && 2 * (lowerWeight - lower.weight) >= total) {
    const top = lower;
    lower = mergeLower(lower.left, lower.right);
    lowerWeight -= top.weight;
    upper = mergeUpper(upper, resetNode(top));
  }
  while (2 * lowerWeight < total) {
    // lowerWeight < total implies upper is non-empty.
    const top = upper as HeapNode;
    upper = mergeUpper(top.left, top.right);
    lowerWeight += top.weight;
    lower = mergeLower(lower, resetNode(top));
  }

  a.lower = lower;
  a.upper = upper;
  a.lowerWeight = lowerWeight;
  a.total = total;
  // total > 0 forces 2 * lowerWeight >= total > 0, so lower is non-empty.
  a.median = (lower as HeapNode).value;
  if (a.memberTail) a.memberTail.next = b.memberHead;
  else a.memberHead = b.memberHead;
  a.memberTail = b.memberTail;
  return a;
}

/**
 * Isotonic regression on indices lo..hi of the transformed coordinates.
 * b[j] = base[j] - P[j]. Every fitted value is forced into the constant box
 * [floor, ceilL] — the per-index bounds are dominated by these under
 * isotonicity, and clamping targets before PAVA preserves the optimum.
 * Optional frozen pin endpoints: left block fixed to floor, right to ceilL —
 * each sentinel outweighs twice the whole segment, so its block value can
 * never move; feasibility guarantees floor <= ceilL so the sentinels never
 * merge into one another.
 */
function solveSegment(
  lo: number,
  hi: number,
  b: ReadonlyArray<number>,
  floor: number,
  ceilL: number,
  leftPin: boolean,
  rightPin: boolean,
): { y: number[]; cost: number } {
  // Every frozen endpoint outweighs twice the whole segment, so its block can
  // never evaluate to anything but its pinned value.
  const pinWeight = 2 * (hi - lo + 1) + 2;
  const stack: Block[] = [];

  const push = (index: number, value: number, weight: number): void => {
    let blk = singleton(index, value, weight);
    while (stack.length > 0) {
      const top = stack[stack.length - 1];
      if (top.median <= blk.median) break;
      stack.pop();
      blk = mergeBlocks(top, blk);
    }
    stack.push(blk);
  };

  // Segment members are clamped to the constant feasible box; a fitted value
  // at a box edge is equivalent in feasibility and the isotonic projection is
  // unchanged because the frozen endpoint sentinels enforce monotonic access.
  if (leftPin) push(lo - 1, floor, pinWeight);
  for (let j = lo; j <= hi; j++) {
    const v = Math.min(ceilL, Math.max(floor, b[j]));
    push(j, v, 1);
  }
  if (rightPin) push(hi + 1, ceilL, pinWeight); // sentinel, output excluded

  const y = new Array<number>(hi - lo + 1);
  let cost = 0;
  for (const blk of stack) {
    const value = blk.median;
    let entry = blk.memberHead;
    while (entry !== null) {
      const j = entry.index;
      if (j >= lo && j <= hi) {
        y[j - lo] = value;
        // Cost is measured against the unclamped target b[j].
        cost += Math.abs(value - b[j]);
      }
      entry = entry.next;
    }
  }
  return { y, cost };
}

export function solve(input: SolveInput): SolveResult {
  return solveCore(input.cues, input.base, input.pins ?? new Map(), DAY_MS);
}

function solveCore(
  cues: ReadonlyArray<Cue>,
  base: ReadonlyArray<number>,
  pins: Pins,
  daySpan: number,
): SolveResult {
  const n = cues.length;
  if (n === 0) return { ok: true, starts: [], cost: 0 };

  // Prefix durations P[i] = sum_{k < i} duration[k].
  const P = new Array<number>(n);
  P[0] = 0;
  for (let i = 0; i + 1 < n; i++) {
    P[i + 1] = P[i] + cues[i].duration;
  }
  const b = new Array<number>(n);
  for (let i = 0; i < n; i++) b[i] = base[i] - P[i];

  const pinAt = new Array<number | null>(n).fill(null);
  for (const [idx, start] of pins) {
    if (
      !Number.isInteger(idx) ||
      idx < 0 ||
      idx >= n ||
      !Number.isInteger(start) ||
      start < 0 ||
      start > daySpan
    ) {
      return { ok: false, reason: 'INFEASIBLE' };
    }
    pinAt[idx] = start;
  }

  // Sorted pin indices.
  const pinIdx: number[] = [];
  for (let i = 0; i < n; i++) if (pinAt[i] !== null) pinIdx.push(i);

  // Feasibility in x-space.
  // Leading / interior / trailing prefix-duration bounds.
  if (pinIdx.length > 0) {
    const first = pinIdx[0];
    if (pinAt[first]! < P[first]) return { ok: false, reason: 'INFEASIBLE' };
    for (let k = 1; k < pinIdx.length; k++) {
      const p = pinIdx[k - 1];
      const q = pinIdx[k];
      if (pinAt[q]! - pinAt[p]! < P[q] - P[p]) {
        return { ok: false, reason: 'INFEASIBLE' };
      }
    }
    const last = pinIdx[pinIdx.length - 1];
    if (pinAt[last]! > daySpan - (P[n - 1] - P[last])) {
      return { ok: false, reason: 'INFEASIBLE' };
    }
  } else if (P[n - 1] > daySpan) {
    return { ok: false, reason: 'INFEASIBLE' };
  }

  // Transformed pin values y = x - P[i].
  const pinY = new Map<number, number>();
  for (const i of pinIdx) pinY.set(i, pinAt[i]! - P[i]);

  const y = new Array<number>(n);
  let cost = 0;

  if (pinIdx.length === 0) {
    // Per-index bounds: y[0] >= 0 and y[n-1] <= daySpan - P[n-1]; all others
    // are dominated by these and the isotonic chain.
    const seg = solveSegment(0, n - 1, b, 0, daySpan - P[n - 1], false, false);
    for (let i = 0; i < n; i++) y[i] = seg.y[i];
    cost = seg.cost;
  } else {
    const first = pinIdx[0];
    if (first > 0) {
      // Leading segment: lower bound 0 dominates all y[i] >= -P[i].
      const seg = solveSegment(
        0,
        first - 1,
        b,
        0,
        pinY.get(first)!,
        false,
        true,
      );
      for (let t = 0; t < first; t++) y[t] = seg.y[t];
      cost += seg.cost;
    }
    for (let k = 0; k < pinIdx.length; k++) {
      const p = pinIdx[k];
      const q = k + 1 < pinIdx.length ? pinIdx[k + 1] : n;
      y[p] = pinY.get(p)!;
      // The pinned cue's own displacement is a constant under the constraint
      // but still belongs to the total absolute displacement.
      cost += Math.abs(pinY.get(p)! - b[p]);
      if (p + 1 <= q - 1) {
        // Trailing: upper bound daySpan - P[n-1] dominates y[i] <= D - P[i].
        const isTrailing = q === n;
        const seg = solveSegment(
          p + 1,
          q - 1,
          b,
          pinY.get(p)!,
          isTrailing ? daySpan - P[n - 1] : pinY.get(q)!,
          true,
          isTrailing ? false : true,
        );
        for (let t = p + 1; t <= q - 1; t++) y[t] = seg.y[t - (p + 1)];
        cost += seg.cost;
      }
    }
  }

  const starts = new Array<number>(n);
  for (let i = 0; i < n; i++) starts[i] = y[i] + P[i];

  // Defensive verification (contract checks; should never fire).
  for (let i = 0; i < n; i++) {
    if (!Number.isInteger(starts[i]) || starts[i] < 0 || starts[i] > daySpan) {
      return { ok: false, reason: 'INFEASIBLE' };
    }
    if (i > 0 && starts[i] < starts[i - 1] + cues[i - 1].duration) {
      return { ok: false, reason: 'INFEASIBLE' };
    }
    if (pinAt[i] !== null && starts[i] !== pinAt[i]) {
      return { ok: false, reason: 'INFEASIBLE' };
    }
  }

  return { ok: true, starts, cost };
}
