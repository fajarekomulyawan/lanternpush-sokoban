# Lanternpush — a push-optimal Sokoban solver

A solver for make.fun's generated Sokoban levels, written for the
"Build the best Sokoban solver, 21 November 2026" bounty. `main.ts` reads each
instance's `level.txt` from `/input/<id>/` and writes its moves (LURD) to
`/output/<id>/solution.txt`.

## How it works

The generated levels are a one-square hallway with small rooms off it, and a
box pushed out of its room can never come back — so a level's push optimum is
the sum of its rooms' optima. The solver:

1. Splits the level at the player's hallway row; the connected floor regions
   above and below it are the rooms.
2. Solves each room with an A\* over pushes (a state is the sorted box squares
   plus the player's reachable region), guided by a minimum-cost matching of
   boxes to goals over precomputed push distances — an admissible estimate, so
   the solutions are push-optimal.
3. Prunes with static dead squares (a box there can reach no goal), a
   fixed-point test for frozen boxes, and the solid-2x2 deadlock rule.

A whole-level push A\* and a plain move search stand behind the room solver as
fallbacks for any level the split does not fit, and every candidate answer is
replayed internally before it is written. All searches are budgeted by
expansions, never by the clock, so two runs write byte-identical output.

On the kit's practice set it solves all five instances in exactly the optimal
push counts (13, 16, 29, 47 and 60) in well under a second.

## Layout

- `main.ts` — the whole solver: parsing, static analysis, the push A*, the
  room split, the fallbacks and the replay check.
- `LICENSE` — MIT.
