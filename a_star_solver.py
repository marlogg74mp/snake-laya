"""
A* pathfinding solver with flood-fill safety verification and tail-chasing fallback for SnakeEnv.
Implements SnakeAStarSolver to guide the snake towards food safely and reliably.
"""

from collections import deque
import heapq
from typing import List, Tuple, Dict, Set, Optional, Any

try:
    from snake_env import ACTIONS, OPPOSITES
except ImportError:
    ACTIONS = {
        "UP": (0, -1),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
        "RIGHT": (1, 0)
    }
    OPPOSITES = {
        "UP": "DOWN",
        "DOWN": "UP",
        "LEFT": "RIGHT",
        "RIGHT": "LEFT"
    }

DELTA_TO_ACTION = {
    (0, -1): "UP",
    (0, 1): "DOWN",
    (-1, 0): "LEFT",
    (1, 0): "RIGHT"
}


class SnakeAStarSolver:
    """
    A* pathfinding solver for SnakeEnv with flood-fill safety checks and tail-chasing fallback.
    """

    def __init__(self):
        pass

    def get_best_move(self, env: Any) -> str:
        """
        Determines the optimal next move for the snake in the given SnakeEnv.

        Args:
            env: An instance of SnakeEnv.

        Returns:
            Action string: one of "UP", "DOWN", "LEFT", "RIGHT".
        """
        head = env.snake[0]
        food = env.food
        width = env.width
        height = env.height
        current_dir = getattr(env, "current_direction", None)

        # 1. Identify all immediately safe moves
        safe_moves = self._get_safe_moves(env)
        if not safe_moves:
            # No safe moves exist; avoid crashing and continue current direction if valid
            return current_dir if current_dir in ACTIONS else "UP"

        if len(safe_moves) == 1:
            # Only one safe move exists; must pick it to avoid immediate collision
            return safe_moves[0][0]

        # 1.5. Check if food is hidden in Fog of War
        is_fog = getattr(env, "fog_of_war", False)
        sight_radius = getattr(env, "sight_radius", 5)
        food_hidden = False
        if food is None:
            food_hidden = True
        elif is_fog:
            dist = abs(head[0] - food[0]) + abs(head[1] - food[1])
            if dist > sight_radius:
                food_hidden = True

        if food_hidden:
            return self._exploration_move(env, safe_moves)

        # 2. Attempt A* search to food
        # Obstacles are walls, current snake body parts, and internal obstacle walls
        snake = getattr(env, "snake", getattr(env, "body", []))
        obstacles = set(snake)
        if hasattr(env, "obstacles"):
            obstacles.update(env.obstacles)
        portals = getattr(env, "portals", [])
        path_to_food = self._a_star_search(head, food, obstacles, width, height, portals=portals)

        if path_to_food and len(path_to_food) > 1:
            first_step = path_to_food[1]
            first_action = self._get_action(head, first_step)

            safe_actions = {m[0] for m in safe_moves}
            if first_action in safe_actions:
                # 3. Safety check: ensure following path does not trap the snake
                if not self._is_path_trapping(env, path_to_food):
                    return first_action

        # 4. Fallback: choose safe direction that maximizes reachable space or moves towards tail
        return self._fallback_move(env, safe_moves)

    def _is_obstacle(self, env: Any, pos: Tuple[int, int]) -> bool:
        """
        Checks whether position is out of bounds or collides with any snake body part.
        """
        if hasattr(env, "is_collision"):
            return env.is_collision(pos)
        if hasattr(env, "is_obstacle"):
            return env.is_obstacle(pos)
        x, y = pos
        if x < 0 or x >= env.width or y < 0 or y >= env.height:
            return True
        snake = getattr(env, "snake", getattr(env, "body", []))
        return pos in snake

    def _get_safe_moves(self, env: Any) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Finds all immediate moves that do not collide with walls, obstacles, or the snake's own body.
        """
        snake = getattr(env, "snake", getattr(env, "body", []))
        head = snake[0]
        current_dir = getattr(env, "current_direction", getattr(env, "direction", None))
        portals = getattr(env, "portals", [])
        safe = []

        for action, (dx, dy) in ACTIONS.items():
            # Disallow instant 180-degree reversal if snake length > 1
            if len(snake) > 1 and current_dir and action == OPPOSITES.get(current_dir):
                continue

            next_pos = (head[0] + dx, head[1] + dy)
            if len(portals) >= 2:
                if next_pos == portals[0]:
                    next_pos = portals[1]
                elif next_pos == portals[1]:
                    next_pos = portals[0]

            if not self._is_obstacle(env, next_pos):
                safe.append((action, next_pos))

        return safe

    def _get_action(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
        """
        Translates a single grid step into a directional action string.
        """
        delta = (to_pos[0] - from_pos[0], to_pos[1] - from_pos[1])
        return DELTA_TO_ACTION.get(delta, "UP")

    def _a_star_search(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
        width: int,
        height: int,
        portals: Optional[List[Tuple[int, int]]] = None
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Executes A* shortest-path search from start to goal avoiding obstacles and using portals.
        Returns coordinate path [start, ..., goal] or None if no path exists.
        """
        if start == goal:
            return [start]

        def heuristic(pos: Tuple[int, int]) -> int:
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        # Priority queue entries: (f_score, g_score, counter, current_pos)
        counter = 0
        open_heap = []
        heapq.heappush(open_heap, (heuristic(start), 0, counter, start))

        g_score: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed_set: Set[Tuple[int, int]] = set()

        while open_heap:
            _, current_g, _, current = heapq.heappop(open_heap)

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            cx, cy = current
            for dx, dy in ACTIONS.values():
                nx, ny = cx + dx, cy + dy
                neighbor = (nx, ny)

                if portals and len(portals) >= 2:
                    if neighbor == portals[0]:
                        neighbor = portals[1]
                    elif neighbor == portals[1]:
                        neighbor = portals[0]

                # Grid boundary check
                if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                    continue

                # Obstacle check (permit goal even if in obstacles, e.g. when chasing tail)
                if neighbor in obstacles and neighbor != goal:
                    continue

                tentative_g = current_g + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    came_from[neighbor] = current
                    counter += 1
                    heapq.heappush(open_heap, (f_score, tentative_g, counter, neighbor))

        return None

    def _flood_fill(
        self,
        start: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
        width: int,
        height: int
    ) -> int:
        """
        Calculates the number of reachable connected cells using breadth-first search.
        """
        if not (0 <= start[0] < width and 0 <= start[1] < height):
            return 0

        # Start position itself is the origin and should not be considered an obstacle to itself
        effective_obstacles = obstacles - {start}

        visited = {start}
        queue = deque([start])

        while queue:
            cx, cy = queue.popleft()
            for dx, dy in ACTIONS.values():
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height:
                    neighbor = (nx, ny)
                    if neighbor not in effective_obstacles and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

        return len(visited)

    def _simulate_path(
        self,
        snake: List[Tuple[int, int]],
        path: List[Tuple[int, int]],
        food: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Simulates snake state after traversing path until food is consumed.
        """
        virtual_snake = list(snake)
        for step in path[1:]:
            virtual_snake.insert(0, step)
            if step != food:
                virtual_snake.pop()
            else:
                # Eating food increases length (tail is preserved)
                break
        return virtual_snake

    def _is_path_trapping(self, env: Any, path_to_food: List[Tuple[int, int]]) -> bool:
        """
        Evaluates whether following the path to food traps the snake in a confined space.
        """
        first_step = path_to_food[1]
        snake_len = len(env.snake)
        total_cells = env.width * env.height
        total_free = max(1, total_cells - snake_len)
        threshold = min(snake_len, total_free)

        # Check immediate safety after the first step
        first_snake = [first_step] + (list(env.snake) if first_step == env.food else list(env.snake[:-1]))
        first_space = self._flood_fill(first_step, set(first_snake), env.width, env.height)
        if first_space < threshold:
            return True

        # Check safety after reaching food
        virtual_snake = self._simulate_path(env.snake, path_to_food, env.food)
        v_head = virtual_snake[0]
        v_tail = virtual_snake[-1]
        v_obstacles = set(virtual_snake)

        food_space = self._flood_fill(v_head, v_obstacles, env.width, env.height)
        if food_space < threshold:
            # If reachable space is constrained, check if snake can reach its tail as escape route
            can_reach_tail = self._a_star_search(v_head, v_tail, v_obstacles, env.width, env.height) is not None
            if not can_reach_tail:
                return True

        return False

    def _exploration_move(self, env: Any, safe_moves: List[Tuple[str, Tuple[int, int]]]) -> str:
        """
        Exploration policy for Fog of War: sweeps across arena, maintains momentum,
        spreads away from tail, and avoids looping.
        """
        width = env.width
        height = env.height
        snake = env.snake
        snake_len = len(snake)
        current_dir = getattr(env, "current_direction", getattr(env, "direction", "RIGHT"))

        evaluated = []
        for action, next_pos in safe_moves:
            next_snake = [next_pos] + list(snake[:-1])
            obstacles = set(next_snake)
            if hasattr(env, "obstacles"):
                obstacles.update(env.obstacles)

            space = self._flood_fill(next_pos, obstacles, width, height)
            next_tail = next_snake[-1]
            tail_path = self._a_star_search(next_pos, next_tail, obstacles, width, height)
            can_reach_tail = tail_path is not None

            # Discard moves with severely constrained space unless tail is reachable
            if space < min(snake_len, 10) and not can_reach_tail:
                continue

            # Momentum bonus: prioritize going straight to sweep arena
            momentum = 40 if action == current_dir else 0

            # Tail spreading for short snakes (prevents 2x2 looping!)
            tail_dist = abs(next_pos[0] - next_tail[0]) + abs(next_pos[1] - next_tail[1])
            if snake_len <= 8:
                tail_score = tail_dist * 15
            else:
                tail_score = 30 if can_reach_tail else 0

            space_score = min(space, 40) * 2

            total_score = momentum + tail_score + space_score
            evaluated.append({"action": action, "score": total_score})

        if not evaluated:
            return self._fallback_move(env, safe_moves)

        evaluated.sort(key=lambda m: m["score"], reverse=True)
        return evaluated[0]["action"]

    def _fallback_move(self, env: Any, safe_moves: List[Tuple[str, Tuple[int, int]]]) -> str:
        """
        Fallback when food path is missing or dangerous:
        Picks a safe move that maximizes reachable flood-fill space or moves towards tail.
        """
        width = env.width
        height = env.height
        snake_len = len(env.snake)

        evaluated = []
        for action, next_pos in safe_moves:
            next_snake = [next_pos] + (list(env.snake) if next_pos == env.food else list(env.snake[:-1]))
            obstacles = set(next_snake)

            space = self._flood_fill(next_pos, obstacles, width, height)
            next_tail = next_snake[-1]

            tail_path = self._a_star_search(next_pos, next_tail, obstacles, width, height)
            can_reach_tail = tail_path is not None
            tail_dist = abs(next_pos[0] - next_tail[0]) + abs(next_pos[1] - next_tail[1])
            food_dist = abs(next_pos[0] - env.food[0]) + abs(next_pos[1] - env.food[1])

            evaluated.append({
                "action": action,
                "next_pos": next_pos,
                "space": space,
                "can_reach_tail": can_reach_tail,
                "tail_dist": tail_dist,
                "food_dist": food_dist
            })

        # Separate moves that retain a clear path to tail
        moves_reaching_tail = [m for m in evaluated if m["can_reach_tail"]]

        if moves_reaching_tail:
            # Among moves reaching tail, prioritize having sufficient space (>= snake length),
            # then maximize open space and move towards tail / food
            best = max(
                moves_reaching_tail,
                key=lambda m: (
                    m["space"] >= snake_len,
                    m["space"],
                    -m["tail_dist"],
                    -m["food_dist"]
                )
            )
        else:
            # If tail cannot be reached, maximize open space to survive as long as possible
            best = max(
                evaluated,
                key=lambda m: (
                    m["space"],
                    -m["tail_dist"],
                    -m["food_dist"]
                )
            )

        return best["action"]
