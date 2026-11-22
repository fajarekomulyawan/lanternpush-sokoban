// Lanternpush — a push-optimal Sokoban solver for make.fun's generated levels.
//
// The generated levels are a one-square hallway with small rooms off it, and a
// box pushed out of its room can never come back, so the push optimum of a
// level is the sum of its rooms' optima. This solver therefore splits a level
// into the rooms off the player's hallway row and solves each room on its own
// with an A* over pushes, which is fast and exact for rooms of three or four
// boxes; a whole-level push A* and a plain move search stand behind it as
// fallbacks for any level the split does not fit. Every search is budgeted by
// expansions, never by the clock, so two runs write the same bytes.
//
// A push state is the sorted box squares plus the player's reachable region
// (named by its smallest square). The estimate is a minimum-cost matching of
// boxes to goals over precomputed push distances, which is admissible, so the
// A* returns push-optimal solutions. Static dead squares (a box there can
// reach no goal) and frozen boxes (a box no push can ever move again) prune
// the search.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const DIRS: [number, number, string][] = [
  [0, -1, 'u'],
  [0, 1, 'd'],
  [-1, 0, 'l'],
  [1, 0, 'r'],
];
const INF = 1_000_000_000;
// One work budget for the whole run, spent by expansions, so the set's later
// instances still get their turn whatever the earlier ones cost.
let workLeft = 1_400_000;

// ---------- the level ----------

interface Level {
  w: number;
  n: number;
  wall: Uint8Array;
  goal: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
  /** dist[g][cell]: pushes to bring a box from cell to goals[g], boxes ignored; -1 never. */
  dist: Int32Array[];
  /** Cells (not goals) from which no goal is reachable by pushes. */
  dead: Uint8Array;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const w = Math.max(...rows.map((r) => r.length));
  const n = w * rows.length;
  const wall = new Uint8Array(n).fill(1);
  const goal = new Uint8Array(n);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') {
        goal[i] = 1;
        goals.push(i);
      }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  // Push distances to each goal: breadth-first over box squares, walking the
  // push graph backwards (a box at c was pushed there from c - d by a player
  // standing at c - 2d).
  const dist = goals.map((g) => {
    const d = new Int32Array(n).fill(-1);
    d[g] = 0;
    const queue = [g];
    for (let head = 0; head < queue.length; head++) {
      const c = queue[head]!;
      const cx = c % w;
      const cy = (c / w) | 0;
      for (const [dx, dy] of DIRS) {
        const bx = cx - dx;
        const by = cy - dy;
        const sx = cx - 2 * dx;
        const sy = cy - 2 * dy;
        if (bx < 0 || by < 0 || bx >= w || by >= rows.length || sx < 0 || sy < 0 || sx >= w || sy >= rows.length) continue;
        const b = by * w + bx;
        if (wall[b] || wall[sy * w + sx] || d[b] >= 0) continue;
        d[b] = d[c]! + 1;
        queue.push(b);
      }
    }
    return d;
  });
  const dead = new Uint8Array(n);
  for (let c = 0; c < n; c++) {
    if (wall[c] || goal[c]) continue;
    if (dist.every((d) => d[c]! < 0)) dead[c] = 1;
  }
  return { w, n, wall, goal, goals, boxes, player, dist, dead };
}

// ---------- small tools ----------

/** Minimum-cost matching of rows to columns of a square cost matrix (Hungarian method). */
function assignment(cost: number[][]): number {
  const n = cost.length;
  if (n === 0) return 0;
  const u = new Float64Array(n + 1);
  const v = new Float64Array(n + 1);
  const p = new Int32Array(n + 1);
  const way = new Int32Array(n + 1);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(n + 1).fill(Infinity);
    const used = new Uint8Array(n + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0]!;
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const cur = cost[i0 - 1]![j - 1]! - u[i0]! - v[j]!;
        if (cur < minv[j]!) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j]! < delta) {
          delta = minv[j]!;
          j1 = j;
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]!] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0]!;
      p[j0] = p[j1]!;
      j0 = j1;
    } while (j0 !== 0);
  }
  let total = 0;
  for (let j = 1; j <= n; j++) {
    if (p[j] === 0) return INF;
    total += cost[p[j]! - 1]![j - 1]!;
    if (total >= INF) return INF;
  }
  return total;
}

/** The squares the player can walk to from start without moving a box, with walk parents. */
function flood(lv: Level, boxSet: Set<number>, start: number): { seen: Uint8Array; parent: Int32Array } {
  const seen = new Uint8Array(lv.n);
  const parent = new Int32Array(lv.n).fill(-1);
  const queue = [start];
  seen[start] = 1;
  for (let head = 0; head < queue.length; head++) {
    const c = queue[head]!;
    const cx = c % lv.w;
    const cy = (c / lv.w) | 0;
    for (const [dx, dy] of DIRS) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (nx < 0 || ny < 0 || nx >= lv.w) continue;
      const q = ny * lv.w + nx;
      if (q >= lv.n || seen[q] || lv.wall[q] || boxSet.has(q)) continue;
      seen[q] = 1;
      parent[q] = c;
      queue.push(q);
    }
  }
  return { seen, parent };
}

/** The walk letters from start to target along a flood's parents. */
function walkPath(lv: Level, parent: Int32Array, start: number, target: number): string {
  let out = '';
  for (let c = target; c !== start; c = parent[c]!) {
    const p = parent[c]!;
    const d = c - p;
    out = (d === 1 ? 'r' : d === -1 ? 'l' : d === lv.w ? 'd' : 'u') + out;
  }
  return out;
}

/**
 * A checker for frozen boxes, built as a fixed point: a box joins the frozen
 * set when every push of it is blocked by a wall or by a box already in the
 * set, and the set grows until nothing more joins. The earliest box of the set
 * to move in any solution would need, for that first push, a landing square
 * and a standing square clear of walls and of the boxes frozen before it —
 * which its joining ruled out — so a frozen box truly never moves again, and
 * one off its goal makes the state a deadlock. (A box outside the set counts
 * as no obstacle: it may move away first.)
 */
function frozenChecker(lv: Level, boxSet: Set<number>): (b: number) => boolean {
  const frozenSet = new Set<number>();
  let grew = true;
  while (grew) {
    grew = false;
    for (const b of boxSet) {
      if (frozenSet.has(b)) continue;
      const bx = b % lv.w;
      const by = (b / lv.w) | 0;
      let movable = false;
      for (const [dx, dy] of DIRS) {
        const dest = (by + dy) * lv.w + bx + dx;
        const stand = (by - dy) * lv.w + bx - dx;
        if (lv.wall[dest] === 1 || frozenSet.has(dest)) continue;
        if (lv.wall[stand] === 1 || frozenSet.has(stand)) continue;
        movable = true;
        break;
      }
      if (!movable) {
        frozenSet.add(b);
        grew = true;
      }
    }
  }
  return (b: number) => frozenSet.has(b);
}

/**
 * Whether the box just pushed to dest completes a solid 2x2 of walls and
 * boxes with some box off its goal. No box of such a block can move first —
 * every push of one has its landing or its standing square inside the block,
 * still occupied — so the state is a deadlock.
 */
function blockDeadlock(lv: Level, boxSet: Set<number>, dest: number): boolean {
  const dx = dest % lv.w;
  const dy = (dest / lv.w) | 0;
  for (const ox of [-1, 0]) {
    for (const oy of [-1, 0]) {
      let solid = true;
      let offGoal = false;
      for (const cx of [0, 1]) {
        for (const cy of [0, 1]) {
          const c = (dy + oy + cy) * lv.w + dx + ox + cx;
          if (lv.wall[c] === 1) continue;
          if (boxSet.has(c)) {
            if (lv.goal[c] !== 1) offGoal = true;
            continue;
          }
          solid = false;
        }
      }
      if (solid && offGoal) return true;
    }
  }
  return false;
}

// ---------- the push A* ----------

interface PushOptions {
  /** Indices into lv.boxes that this search may move; the rest are fixed obstacles. */
  movable: number[];
  /** Goal cells this search must cover (indices into lv.goals give their dist rows). */
  goalIdx: number[];
  /** When set, a push's landing square must be in it (keeps a room's boxes home). */
  home: Set<number> | null;
  /** When set, the final position must leave the player able to reach one of these cells. */
  exitTo: Set<number> | null;
  useFreeze: boolean;
  cap: number;
}

interface PushResult {
  moves: string;
  boxes: number[];
  player: number;
}

/** A* over pushes. Returns the move string and the end position, or null. */
function pushSearch(lv: Level, startBoxes: number[], startPlayer: number, opts: PushOptions): PushResult | null {
  const goalCells = opts.goalIdx.map((gi) => lv.goals[gi]!);
  const goalSet = new Set(goalCells);
  const movableCells = new Set(opts.movable.map((bi) => startBoxes[bi]!));
  const fixed = new Set<number>();
  startBoxes.forEach((b, bi) => {
    if (!opts.movable.includes(bi)) fixed.add(b);
  });
  // Squares a movable box may not rest on: unreachable to every goal of this search.
  const localDead = new Uint8Array(lv.n);
  for (let c = 0; c < lv.n; c++) {
    if (lv.wall[c] || lv.goal[c]) continue;
    if (opts.goalIdx.every((gi) => lv.dist[gi]![c]! < 0)) localDead[c] = 1;
  }
  const estimate = (cells: number[]): number => {
    const cost = cells.map((b) =>
      opts.goalIdx.map((gi) => {
        const d = lv.dist[gi]![b]!;
        return d < 0 ? INF : d;
      }),
    );
    return assignment(cost);
  };
  const startCells = opts.movable.map((bi) => startBoxes[bi]!).sort((a, b) => a - b);
  const startSet = new Set(startBoxes);
  const startFlood = flood(lv, startSet, startPlayer);
  let startCanon = startPlayer;
  for (let c = 0; c < lv.n; c++) if (startFlood.seen[c]) { startCanon = c; break; }

  // Node store: movable box squares, actual player square, its region's name,
  // pushes so far, parent and the move segment that made it.
  const nb: number[][] = [startCells];
  const np: number[] = [startPlayer];
  const nc: number[] = [startCanon];
  const ng: number[] = [0];
  const nparent: number[] = [-1];
  const nseg: string[] = [''];
  const keyOf = (canon: number, cells: number[]): string => String.fromCharCode(canon, ...cells);
  const best = new Map<string, number>([[keyOf(startCanon, startCells), 0]]);
  // The open list: a heap on (f, deeper first, older first) for a stable order.
  const heap: [number, number, number][] = [];
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  const h0 = estimate(startCells);
  if (h0 >= INF) return null;
  if (opts.useFreeze) {
    const checkStart = frozenChecker(lv, startSet);
    for (const c of startCells) if (!lv.goal[c] && checkStart(c)) return null;
  }
  heapPush([h0, 0, 0]);
  let expanded = 0;
  while (heap.length > 0) {
    const [, , s] = heapPop();
    const cells = nb[s]!;
    // best holds the cheapest node for a key; any other node with it is stale.
    if (best.get(keyOf(nc[s]!, cells)) !== s) continue;
    if (cells.every((b) => goalSet.has(b))) {
      if (opts.exitTo !== null) {
        const all = new Set([...cells, ...fixed]);
        const fl = flood(lv, all, np[s]!);
        let out = false;
        for (const c of opts.exitTo) if (fl.seen[c]) { out = true; break; }
        if (!out) {
          // Solved but the player is shut in: keep searching for a way out.
          if (expanded >= opts.cap || workLeft <= 0) return null;
        } else {
          return finish(s);
        }
      } else {
        return finish(s);
      }
    }
    if (expanded >= opts.cap || workLeft <= 0) return null;
    expanded++;
    workLeft--;
    const allBoxes = new Set([...cells, ...fixed]);
    const fl = flood(lv, allBoxes, np[s]!);
    for (let bi = 0; bi < cells.length; bi++) {
      const b = cells[bi]!;
      const bx = b % lv.w;
      const by = (b / lv.w) | 0;
      for (const [dx, dy, letter] of DIRS) {
        const stand = (by - dy) * lv.w + bx - dx;
        const dest = (by + dy) * lv.w + bx + dx;
        if (!fl.seen[stand]) continue;
        if (lv.wall[dest] || allBoxes.has(dest)) continue;
        if (localDead[dest] && !lv.goal[dest]) continue;
        if (opts.home !== null && !opts.home.has(dest)) continue;
        const next = cells.slice();
        next[bi] = dest;
        next.sort((a, b2) => a - b2);
        const nextAll = new Set([...next, ...fixed]);
        if (blockDeadlock(lv, nextAll, dest)) continue;
        if (opts.useFreeze) {
          const check = frozenChecker(lv, nextAll);
          let dead = false;
          for (const c of next) {
            if (!lv.goal[c] && check(c)) { dead = true; break; }
          }
          if (dead) continue;
        }
        const h = estimate(next);
        if (h >= INF) continue;
        // The player's new region, named by its smallest square.
        const nfl = flood(lv, nextAll, b);
        let canon = b;
        for (let c = 0; c < lv.n; c++) if (nfl.seen[c]) { canon = c; break; }
        const k = keyOf(canon, next);
        const g = ng[s]! + 1;
        const seenAt = best.get(k);
        if (seenAt !== undefined && ng[seenAt]! <= g) continue;
        const t = nb.length;
        nb.push(next);
        np.push(b);
        nc.push(canon);
        ng.push(g);
        nparent.push(s);
        nseg.push(walkPath(lv, fl.parent, np[s]!, stand) + letter.toUpperCase());
        best.set(k, t);
        heapPush([g + h, g, t]);
      }
    }
  }
  return null;

  function finish(s: number): PushResult {
    let moves = '';
    for (let t = s; t > 0; t = nparent[t]!) moves = nseg[t] + moves;
    const boxes = startBoxes.slice();
    opts.movable.forEach((bi, i) => {
      boxes[bi] = nb[s]![i]!;
    });
    return { moves, boxes, player: np[s]! };
  }
}

// ---------- replay (the safety net: never write a solution that does not solve) ----------

function replay(lv: Level, moves: string): boolean {
  const box = new Uint8Array(lv.n);
  for (const b of lv.boxes) box[b] = 1;
  let player = lv.player;
  for (const ch of moves) {
    const low = ch.toLowerCase();
    const dir = DIRS.find((d) => d[2] === low);
    if (!dir) return false;
    const d = dir[1] * lv.w + dir[0];
    const next = player + d;
    if (lv.wall[next]) return false;
    if (box[next]) {
      const beyond = next + d;
      if (lv.wall[beyond] || box[beyond]) return false;
      box[next] = 0;
      box[beyond] = 1;
    }
    player = next;
  }
  for (let c = 0; c < lv.n; c++) if (box[c] && !lv.goal[c]) return false;
  return true;
}

// ---------- the room split ----------

/**
 * Split the level at the player's hallway row: the connected floor regions
 * above and below it are the rooms. Returns each room's box and goal indices,
 * or null when the level does not have that shape.
 */
function splitRooms(lv: Level): { rooms: { boxes: number[]; goals: number[]; cells: Set<number> }[]; hallway: Set<number> } | null {
  const row = (lv.player / lv.w) | 0;
  const hallway = new Set<number>();
  for (let x = 0; x < lv.w; x++) {
    const c = row * lv.w + x;
    if (!lv.wall[c]) hallway.add(c);
  }
  const comp = new Int32Array(lv.n).fill(-1);
  let count = 0;
  for (let c = 0; c < lv.n; c++) {
    if (lv.wall[c] || hallway.has(c) || comp[c] >= 0) continue;
    const queue = [c];
    comp[c] = count;
    for (let head = 0; head < queue.length; head++) {
      const cur = queue[head]!;
      const cx = cur % lv.w;
      const cy = (cur / lv.w) | 0;
      for (const [dx, dy] of DIRS) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= lv.w) continue;
        const q = ny * lv.w + nx;
        if (q >= lv.n || lv.wall[q] || hallway.has(q) || comp[q] >= 0) continue;
        comp[q] = count;
        queue.push(q);
      }
    }
    count++;
  }
  const rooms: { boxes: number[]; goals: number[]; cells: Set<number> }[] = [];
  const byComp = new Map<number, { boxes: number[]; goals: number[]; cells: Set<number> }>();
  for (let c = 0; c < lv.n; c++) {
    if (comp[c] < 0) continue;
    let room = byComp.get(comp[c]);
    if (!room) {
      room = { boxes: [], goals: [], cells: new Set() };
      byComp.set(comp[c], room);
    }
    room.cells.add(c);
  }
  lv.boxes.forEach((b, bi) => {
    const room = byComp.get(comp[b]);
    if (!room) return;
    room.boxes.push(bi);
  });
  lv.goals.forEach((g, gi) => {
    const room = byComp.get(comp[g]);
    if (!room) return;
    room.goals.push(gi);
  });
  for (const room of byComp.values()) {
    if (room.boxes.length === 0 && room.goals.length === 0) continue;
    if (room.boxes.length !== room.goals.length || room.boxes.length === 0) return null;
    rooms.push(room);
  }
  if (rooms.length === 0) return null;
  if (rooms.reduce((a, r) => a + r.boxes.length, 0) !== lv.boxes.length) return null;
  if (rooms.reduce((a, r) => a + r.goals.length, 0) !== lv.goals.length) return null;
  rooms.sort((a, b) => Math.min(...a.cells) - Math.min(...b.cells));
  return { rooms, hallway };
}

/** Solve room by room, threading the boxes and the player through. */
function solveByRooms(lv: Level, useFreeze: boolean): string | null {
  const split = splitRooms(lv);
  if (split === null) return null;
  let boxes = lv.boxes.slice();
  let player = lv.player;
  let moves = '';
  for (let ri = 0; ri < split.rooms.length; ri++) {
    const room = split.rooms[ri]!;
    const last = ri === split.rooms.length - 1;
    const res = pushSearch(lv, boxes, player, {
      movable: room.boxes,
      goalIdx: room.goals,
      home: room.cells,
      exitTo: last ? null : split.hallway,
      useFreeze,
      cap: 600_000,
    });
    if (res === null) return null;
    moves += res.moves;
    boxes = res.boxes;
    player = res.player;
  }
  return moves;
}

// ---------- a plain move search, the last resort ----------

/** Breadth-first A* over single steps, weighted to prefer pushes few and moves short. */
function moveSearch(lv: Level, cap: number): string | null {
  const startSet = new Set(lv.boxes);
  const key = (p: number, bs: number[]): string => String.fromCharCode(p, ...bs);
  const sp: number[] = [lv.player];
  const sb: number[][] = [lv.boxes.slice()];
  const sg: number[] = [0];
  const parent: number[] = [-1];
  const step: string[] = [''];
  const best = new Map<string, number>([[key(lv.player, lv.boxes), 0]]);
  // A min-heap on the cost so far: cheapest node first, older first on a tie.
  const heap: [number, number][] = [[0, 0]];
  const heapPush = (e: [number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (heap[i]![0] >= heap[p]![0]) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heap[c + 1]![0] < heap[c]![0]) c++;
        if (heap[c]![0] >= heap[i]![0]) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  let expanded = 0;
  while (heap.length > 0 && expanded < cap && workLeft > 0) {
    const [, s] = heapPop();
    if (best.get(key(sp[s]!, sb[s]!)) !== s) continue;
    const bs = sb[s]!;
    if (bs.every((b) => lv.goal[b] === 1)) {
      let moves = '';
      for (let t = s; t > 0; t = parent[t]!) moves = step[t] + moves;
      return moves;
    }
    expanded++;
    workLeft--;
    const p = sp[s]!;
    const px = p % lv.w;
    const py = (p / lv.w) | 0;
    for (const [dx, dy, letter] of DIRS) {
      const q = (py + dy) * lv.w + px + dx;
      if (lv.wall[q]) continue;
      let nb = bs;
      let cost = 1;
      const at = bs.indexOf(q);
      if (at >= 0) {
        const beyond = q + dy * lv.w + dx;
        if (lv.wall[beyond] || bs.includes(beyond)) continue;
        if (lv.dead[beyond]) continue;
        nb = bs.slice();
        nb[at] = beyond;
        nb.sort((a, b) => a - b);
        cost = 1025;
      }
      const k = key(q, nb);
      const g = sg[s]! + cost;
      const seen = best.get(k);
      if (seen !== undefined && sg[seen]! <= g) continue;
      const t = sp.length;
      sp.push(q);
      sb.push(nb);
      sg.push(g);
      parent.push(s);
      step.push(at >= 0 ? letter.toUpperCase() : letter);
      best.set(k, t);
      heapPush([g, t]);
    }
  }
  return null;
}

// ---------- one level, then the set ----------

function solveLevel(text: string): string {
  const lv = parseLevel(text);
  if (lv.boxes.length === 0) return '';
  const attempts: (string | null)[] = [];
  attempts.push(solveByRooms(lv, true));
  if (attempts[0] === null) attempts.push(solveByRooms(lv, false));
  const allIdx = lv.boxes.map((_, i) => i);
  const allGoals = lv.goals.map((_, i) => i);
  if (attempts.every((a) => a === null)) {
    attempts.push(
      pushSearch(lv, lv.boxes, lv.player, { movable: allIdx, goalIdx: allGoals, home: null, exitTo: null, useFreeze: true, cap: 350_000 })?.moves ?? null,
    );
  }
  if (attempts.every((a) => a === null)) {
    attempts.push(
      pushSearch(lv, lv.boxes, lv.player, { movable: allIdx, goalIdx: allGoals, home: null, exitTo: null, useFreeze: false, cap: 200_000 })?.moves ?? null,
    );
  }
  if (attempts.every((a) => a === null)) attempts.push(moveSearch(lv, 200_000));
  for (const a of attempts) {
    if (a !== null && replay(lv, a)) return a;
  }
  return '';
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
