# Human vs AI — Versus Rules

Builds on the single-snake rules in `RULES_V9.md` (drones, phase walls, immortality, state schema).

## Match
- 20×20 arena, two snakes: **Human** (cyan, starts on the left, heading right) and **Laya** (magenta,
  starts on the right, heading left), both length 3.
- The map is mirror-symmetric (x ↔ 19−x): walls, portals, drone lanes — no side has an advantage.
- One match = **600 steps** (≈ 90 s at 150 ms per step). Both snakes move simultaneously each step.
- Highest score at the end wins; equal score = draw.

## Food and score
| Item | Effect |
|---|---|
| 🍎 apple | +1, +1 length (two are always on the map) |
| 🍏 golden | +3, +1 length (sometimes, lives 60 steps) |
| ⭐ immortality | +2, +1 length, 50 steps of immortality (rare, lives 60 steps) |
| ☠️ poison | deadly unless immortal (sometimes, lives 60 steps) |

## Collisions (checked on the new head cells, after both snakes chose their move)
- Border, solid wall, own body, drone, poison → death (immortality protects from everything except
  border and solid walls).
- Head into the **opponent's body** → the attacker dies, the opponent gets **+5** (kill bonus).
- Heads meet (same cell or swap) → both die; if exactly one is immortal, only the other dies.
- Death: **−3** score (not below 0), the snake disappears and respawns 10 steps later at length 3 in the
  safest free spot with 10 steps of spawn immortality.
- Drones bounce off solid walls and kill on any body contact, as in V9.

## Fog of war
- Every 120–200 steps the fog falls for 50–70 steps (warning 10 steps before), for both players.
- In fog only food within 5 cells (Manhattan) of your own head is visible; walls, drones and the opponent
  stay visible (dimmed). This is exactly how the model was trained with `fog_of_war` (food_dir = UNKNOWN).

## Cameras (3D)
Overview (default) → top-down → chase cam behind your snake (key V or the 📷 button). In the chase cam
controls are relative: left/right = turn left/right, up/down do nothing.

## The AI's view
Laya plays with the V9 model unchanged. The opponent's cells are added to `obstacles` and to the danger
flags, plus the cell in front of the opponent's head (to avoid head-on crashes).
