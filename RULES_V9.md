# Snake Arena — Game Rules V9

Single source of truth for the training simulator (`build_dataset_v9.py`), the classic UI (`web_server.py`)
and the 3D UI (`web3d/js/game.js`). Any rule change must be made in all three, otherwise the model
is trained on one game and plays another.

All durations are counted in **game steps** (one snake move), not seconds, so the model can reason
about them and turbo/freeze orbs do not change the rules. At the normal speed one step ≈ 120 ms + inference.

## 1. Drones (patrols)
- Each drone moves one cell per step along its route (horizontal/vertical ping-pong or rectangular circuit).
- **Solid walls block drones.** If the drone's next cell is a solid wall or outside its route, it reverses
  direction; if the reversed cell is blocked too, it stays in place for that step.
- A drone kills the snake when: the head enters the drone's cell, the head and the drone swap cells,
  or the drone steps onto **any** body segment (`PATROL_BODY_HITS = true`).
- State: `patrols: [[x, y, dx, dy], ...]`, where `(dx, dy)` is the drone's **actual** next step
  (after any bounce), `(0, 0)` if it will stay.

## 2. Phase walls (become transparent for a while)
- Every wall structure (bar, 2x2 block, L-shape) independently cycles:
  `SOLID` (40–80 steps; the first cycle of a new map starts at 15–80) → `BLINKING` (3 steps, still solid, warning) → `GHOST` (15–25 steps) → `SOLID`.
- While `GHOST`, the structure blocks **nobody**: the snake and drones pass through it.
- A ghost wall turns solid again only when none of its cells is occupied by the snake or a drone;
  until then it stays ghost (its counter stays at 1).
- State:
  - `obstacles`: cells of SOLID and BLINKING walls (these are deadly).
  - `phase_walls: [[x, y, steps_left], ...]`: cells of GHOST walls with steps until they become solid,
    plus BLINKING cells with a **negative** `steps_left` = steps until they vanish (e.g. `-2`).

## 3. Immortality apple ⭐
- Spawns rarely (≈8% of food spawns in the UI, ~15% in training), disappears after 60 steps if not eaten.
- Eating it: +2 score, +1 length, and **50 steps of immortality** (≈6–8 s at normal speed).
  Eating another one while immortal resets the counter to 50.
- While immortal the snake is immune to **drones, its own body and poison apples** (poison is eaten
  without effect). **Boundaries and solid walls stay deadly.**
- Being inside its own body or on a drone's cell when immortality ends is not a death by itself;
  only a new collision on the following steps is.
- State: `immortal_steps: N` (0 when not immortal). The `danger_*` flags describe what is deadly
  **right now**: body cells are not dangers while `immortal_steps > 0`, drone cells while `immortal_steps > 1`.
  Drone danger is checked on the raw neighbouring cell (no portal warp); walls and body on the warped cell.

## 4. Order of events in one step
1. The snake moves (portal warp applies). Death if the new head cell is out of bounds, a solid wall,
   or — unless immortal — its own body, a poison apple or a drone's current cell.
2. Food is eaten (immortality apple sets `immortal_steps = 50`).
3. Drones move, using the walls as they were at the start of the step (so the `(dx, dy)` shown in the
   state is exactly what happens). Death — unless immortal — if a drone steps onto any snake cell
   or swaps cells with the head.
4. Wall phases advance (a ghost wall solidifies only if its cells are free).
5. `immortal_steps` decreases by 1 (not on the step the apple was eaten); food lifetimes decrease.

## 5. State schema V9 (key order matters — Laya reads it as text)
```
current_dir, recent_actions, food_dir, danger_UP, danger_DOWN, danger_LEFT, danger_RIGHT,
head_pos, body, food_pos, snake_len, immortal_steps, obstacles, phase_walls, portals,
fog_of_war, sight_radius, patrols, drone_path
```
- `body`: snake cells after the head, neck → tail, at most 30.
- `drone_path`: for each drone (same order as `patrols`), its next 6 cells `[[x, y], ...]`,
  simulated with the walls as they are now (bounces included).
The exact list is saved next to the weights in `state_schema.json`; the server normalizes any
incoming state to it.

## 6. Known simulator quirks (kept so that the UI matches the training data)
- In `build_dataset_v9.py`, food may spawn on a portal cell (it can never be eaten there); the UIs never place food on portals.
- Training data has no poison, golden, shrink or speed items and no loop detector; they exist only in the UIs.
