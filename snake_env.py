"""
Pure Python Snake grid environment for reinforcement learning and dataset generation.
Does not require Pygame or external GUI dependencies.
"""

from typing import List, Tuple, Dict, Any, Optional, Union
import random


class SnakeEnv:
    """
    Pure Python grid environment for the classic Snake game.

    Grid coordinate convention:
    - Top-left is (0, 0).
    - x increases horizontally to the right: [0, width - 1].
    - y increases vertically downwards: [0, height - 1].
    - UP decreases y: (0, -1).
    - DOWN increases y: (0, 1).
    - LEFT decreases x: (-1, 0).
    - RIGHT increases x: (1, 0).
    """

    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

    ACTIONS = [UP, DOWN, LEFT, RIGHT]

    DIRECTIONS: Dict[str, Tuple[int, int]] = {
        UP: (0, -1),
        DOWN: (0, 1),
        LEFT: (-1, 0),
        RIGHT: (1, 0),
    }

    OPPOSITES: Dict[str, str] = {
        UP: DOWN,
        DOWN: UP,
        LEFT: RIGHT,
        RIGHT: LEFT,
    }

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        initial_length: int = 3,
        num_obstacles: int = 0,
        num_portals: int = 2,
        fog_of_war: bool = False,
        sight_radius: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        if width < 4 or height < 4:
            raise ValueError("Grid width and height must be at least 4.")

        self.width = width
        self.height = height
        self.initial_length = initial_length
        self.num_obstacles = num_obstacles
        self.num_portals = num_portals
        self.fog_of_war = fog_of_war
        self.sight_radius = sight_radius
        self.rng = random.Random(seed)

        self.head: Tuple[int, int] = (0, 0)
        self.body: List[Tuple[int, int]] = []
        self.obstacles: List[Tuple[int, int]] = []
        self.portals: List[Tuple[int, int]] = []
        self.food: Optional[Tuple[int, int]] = None
        self.direction: str = self.RIGHT
        self.recent_actions: List[str] = [self.RIGHT] * 5
        self.score: int = 0
        self.steps: int = 0
        self.done: bool = False

        self.reset()

    @property
    def snake(self) -> List[Tuple[int, int]]:
        """Alias for self.body to maintain compatibility with solver scripts."""
        return self.body

    @property
    def current_direction(self) -> str:
        """Alias for self.direction."""
        return self.direction

    @current_direction.setter
    def current_direction(self, val: str) -> None:
        self.direction = val

    def is_obstacle(self, pos: Union[Tuple[int, int], List[int]]) -> bool:
        """Returns True if position is out of bounds, hits snake body, or hits obstacles."""
        return self.is_collision(pos)

    def reset(
        self,
        seed: Optional[int] = None,
        initial_length: Optional[int] = None,
        num_obstacles: Optional[int] = None,
        num_portals: Optional[int] = None,
    ) -> Dict[str, Any]:
        if seed is not None:
            self.rng.seed(seed)

        length = initial_length if initial_length is not None else self.initial_length
        n_obs = num_obstacles if num_obstacles is not None else self.num_obstacles
        n_ports = num_portals if num_portals is not None else self.num_portals

        cx = self.width // 2
        cy = self.height // 2
        self.head = (cx, cy)
        self.direction = self.RIGHT
        self.recent_actions = [self.RIGHT] * 5
        self.score = 0
        self.steps = 0
        self.done = False

        # Build initial body trailing horizontally to the left
        self.body = [(cx - i, cy) for i in range(length)]

        # Generate random multi-cell wall structures (bars, 2x2 blocks, L-shapes)
        self.obstacles = []
        forbidden = set(self.body)
        # Keep 3-cell buffer around head for fair start
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                forbidden.add((cx + dx, cy + dy))

        shape_templates = [
            [(0, 0), (1, 0), (2, 0)],         # Horizontal 3-cell bar
            [(0, 0), (0, 1), (0, 2)],         # Vertical 3-cell bar
            [(0, 0), (1, 0), (0, 1), (1, 1)], # 2x2 Square Block
            [(0, 0), (1, 0), (2, 0), (0, 1)], # L-Shape
            [(0, 0), (1, 0)],                 # Horizontal 2-cell bar
            [(0, 0), (0, 1)],                 # Vertical 2-cell bar
        ]

        for _ in range(n_obs):
            attempts = 0
            while attempts < 100:
                template = self.rng.choice(shape_templates)
                ox = self.rng.randint(1, self.width - 3)
                oy = self.rng.randint(1, self.height - 3)
                
                shape_cells = [(ox + dx, oy + dy) for (dx, dy) in template]
                
                # Check if all cells in shape are valid and free
                valid = all(
                    0 <= x < self.width and 0 <= y < self.height and (x, y) not in forbidden
                    for (x, y) in shape_cells
                )
                
                if valid:
                    for cell in shape_cells:
                        self.obstacles.append(cell)
                        forbidden.add(cell)
                    break
                attempts += 1

        # Generate 1 pair of Portals (Portal A & Portal B) at least 5 units apart
        self.portals = []
        if n_ports >= 2:
            attempts = 0
            while attempts < 200:
                pa = (self.rng.randint(1, self.width - 2), self.rng.randint(1, self.height - 2))
                pb = (self.rng.randint(1, self.width - 2), self.rng.randint(1, self.height - 2))
                dist = abs(pa[0] - pb[0]) + abs(pa[1] - pb[1])
                if pa not in forbidden and pb not in forbidden and dist >= 6:
                    self.portals = [pa, pb]
                    break
                attempts += 1

        # Spawn food on an empty cell
        self.food = self._spawn_food()

        return self.get_compact_state()

    def _spawn_food(self) -> Optional[Tuple[int, int]]:
        occupied_set = set(self.body)
        if hasattr(self, "obstacles"):
            occupied_set.update(self.obstacles)
        total_cells = self.width * self.height

        if len(occupied_set) >= total_cells:
            return None

        # When grid is sparsely occupied, rejection sampling is fast O(1)
        if len(occupied_set) < total_cells * 0.75:
            while True:
                candidate = (
                    self.rng.randint(0, self.width - 1),
                    self.rng.randint(0, self.height - 1),
                )
                if candidate not in occupied_set:
                    return candidate

        # When grid is dense, sample directly from remaining free cells
        free_cells = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in occupied_set
        ]
        return self.rng.choice(free_cells)

    def is_collision(self, point: Union[Tuple[int, int], List[int]]) -> bool:
        """
        Check if a given coordinate collides with walls, obstacles, or any snake body segment.
        """
        x, y = point[0], point[1]
        # Wall collision check
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        # Snake body collision check
        if (x, y) in self.body:
            return True
        # Internal obstacle check
        if hasattr(self, "obstacles") and (x, y) in self.obstacles:
            return True
        return False

    def step(
        self, action: Union[str, int]
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.done:
            info = {"score": self.score, "steps": self.steps, "warning": "Already done"}
            return self.get_compact_state(), 0.0, True, info

        # Parse string action or integer index
        if isinstance(action, int):
            if 0 <= action < len(self.ACTIONS):
                action_str = self.ACTIONS[action]
            else:
                raise ValueError(
                    f"Action index {action} out of range (0-{len(self.ACTIONS) - 1})"
                )
        elif isinstance(action, str):
            action_str = action.strip().upper()
            if action_str not in self.DIRECTIONS:
                raise ValueError(
                    f"Invalid action '{action}'. Valid actions are {self.ACTIONS}"
                )
        else:
            raise TypeError(f"Action must be str or int, got {type(action)}")

        # Prevent instant 180-degree reversal if snake length > 1
        if len(self.body) > 1 and action_str == self.OPPOSITES.get(self.direction):
            action_str = self.direction
        else:
            self.direction = action_str

        if not hasattr(self, "recent_actions"):
            self.recent_actions = [self.direction] * 5
        self.recent_actions.append(action_str)
        if len(self.recent_actions) > 5:
            self.recent_actions.pop(0)

        # Calculate new head position
        dx, dy = self.DIRECTIONS[action_str]
        new_head = (self.head[0] + dx, self.head[1] + dy)

        # Check Portal Warp
        if hasattr(self, "portals") and len(self.portals) >= 2:
            if new_head == self.portals[0]:
                new_head = self.portals[1]
            elif new_head == self.portals[1]:
                new_head = self.portals[0]

        # Collision check with walls, obstacles, self
        hit = self.is_collision(new_head)

        if hit:
            self.done = True
            reward = -10.0
            info = {
                "score": self.score,
                "steps": self.steps,
                "reason": "collision",
            }
            return self.get_compact_state(), reward, True, info

        # Advance snake
        self.steps += 1
        self.head = new_head
        self.body.insert(0, new_head)

        is_eating = (self.food is not None and new_head == self.food)
        if is_eating:
            self.score += 1
            reward = 10.0

            # Check win condition: snake occupies entire board
            if len(self.body) >= self.width * self.height:
                self.food = None
                self.done = True
                info = {"score": self.score, "steps": self.steps, "reason": "win"}
                return self.get_compact_state(), reward, True, info

            # Spawn next food item
            self.food = self._spawn_food()
            info = {"score": self.score, "steps": self.steps}
            return self.get_compact_state(), reward, False, info
        else:
            # Pop tail segment
            self.body.pop()
            reward = 0.0
            info = {"score": self.score, "steps": self.steps}
            return self.get_compact_state(), reward, False, info

    def _get_food_dir(self) -> str:
        if self.food is None:
            return "SAME"

        if getattr(self, "fog_of_war", False):
            dist = abs(self.food[0] - self.head[0]) + abs(self.food[1] - self.head[1])
            if dist > getattr(self, "sight_radius", 5):
                return "UNKNOWN"

        dx = self.food[0] - self.head[0]
        dy = self.food[1] - self.head[1]

        if dx == 0 and dy < 0:
            return "UP"
        if dx == 0 and dy > 0:
            return "DOWN"
        if dx < 0 and dy == 0:
            return "LEFT"
        if dx > 0 and dy == 0:
            return "RIGHT"
        if dx < 0 and dy < 0:
            return "UP_LEFT"
        if dx > 0 and dy < 0:
            return "UP_RIGHT"
        if dx < 0 and dy > 0:
            return "DOWN_LEFT"
        if dx > 0 and dy > 0:
            return "DOWN_RIGHT"
        return "SAME"

    def get_compact_state(self) -> Dict[str, Any]:
        hx, hy = self.head
        
        def check_danger(next_p: Tuple[int, int]) -> bool:
            if hasattr(self, "portals") and len(self.portals) >= 2:
                if next_p == self.portals[0]:
                    next_p = self.portals[1]
                elif next_p == self.portals[1]:
                    next_p = self.portals[0]
            return self.is_collision(next_p)

        is_fog = getattr(self, "fog_of_war", False)
        s_radius = getattr(self, "sight_radius", 5)

        food_visible = True
        if self.food is None:
            food_visible = False
        elif is_fog:
            dist = abs(self.food[0] - hx) + abs(self.food[1] - hy)
            if dist > s_radius:
                food_visible = False

        food_coord = [self.food[0], self.food[1]] if (self.food and food_visible) else [-1, -1]

        return {
            "current_dir": self.direction,
            "recent_actions": list(getattr(self, "recent_actions", [self.direction] * 5)),
            "food_dir": self._get_food_dir(),
            "danger_UP": check_danger((hx, hy - 1)),
            "danger_DOWN": check_danger((hx, hy + 1)),
            "danger_LEFT": check_danger((hx - 1, hy)),
            "danger_RIGHT": check_danger((hx + 1, hy)),
            "head_pos": [hx, hy],
            "food_pos": food_coord,
            "obstacles": [[ox, oy] for (ox, oy) in getattr(self, "obstacles", [])],
            "portals": [[px, py] for (px, py) in getattr(self, "portals", [])],
            "fog_of_war": is_fog,
            "sight_radius": s_radius
        }

    def render(self, mode: str = "ascii") -> str:
        """
        Render the current state of the board in ASCII.

        Characters:
            'H': Snake head
            'O': Snake body segment
            '*': Food item
            '.': Empty cell
        """
        board = [["." for _ in range(self.width)] for _ in range(self.height)]

        if self.food is not None:
            fx, fy = self.food
            if 0 <= fx < self.width and 0 <= fy < self.height:
                board[fy][fx] = "*"

        for segment in self.body[1:]:
            bx, by = segment
            if 0 <= bx < self.width and 0 <= by < self.height:
                board[by][bx] = "O"

        hx, hy = self.head
        if 0 <= hx < self.width and 0 <= hy < self.height:
            board[hy][hx] = "H"

        lines = ["#" * (self.width + 2)]
        for row in board:
            lines.append("#" + "".join(row) + "#")
        lines.append("#" * (self.width + 2))
        rendered_str = "\n".join(lines)

        if mode == "human":
            print(rendered_str)
        return rendered_str
