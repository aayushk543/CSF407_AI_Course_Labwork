"""
Warehouse Navigation Agent
==========================

This program implements a GOAL-BASED AGENT that plans a collision-free
path for an autonomous warehouse vehicle, from a start position 'S' to
a goal position 'G', on a 2-D grid warehouse map. Obstacles ('#') may
not be crossed; free space ('.') may be crossed.

Goal-based agent architecture
------------------------------
    ENVIRONMENT   -> the warehouse grid (static, fully observable)
    STATE         -> the vehicle's current (row, col) position
    GOAL          -> reach the grid cell marked 'G'
    ACTIONS       -> Up, Down, Left, Right (each moves one cell)
    DECISION      -> a search procedure (Breadth-First Search, BFS) that
                      looks ahead through possible future states and
                      chooses the sequence of actions that reaches the
                      goal, rather than reacting to the current percept
                      alone. This is what makes it "goal-based" instead
                      of a simple reflex agent (see PERCEPT_SEQUENCE
                      discussion in the accompanying report).

Why BFS?
--------
Every move has the same cost (one grid step), so the problem is an
*unweighted* shortest-path problem. On an unweighted graph, Breadth-
First Search is guaranteed to find a path with the minimum number of
moves, and it does so systematically by exploring the search space in
order of increasing distance from the start. It is simple to implement,
easy to reason about, and optimal for this exact class of problem.
(An alternative such as A* would also work and would generally explore
fewer nodes by using a heuristic such as Manhattan distance, but for a
grid of this size the difference in efficiency is negligible.)
"""

from collections import deque

# ---------------------------------------------------------------------
# 1. THE ENVIRONMENT: the warehouse map
# ---------------------------------------------------------------------
WAREHOUSE_MAP = [
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################",
]

# The four actions available to the agent, and the (row, col) delta
# that each action produces.
ACTIONS = {
    "Up":    (-1, 0),
    "Down":  (1, 0),
    "Left":  (0, -1),
    "Right": (0, 1),
}


def find_symbol(grid, symbol):
    """Locate the (row, col) of a given symbol in the grid."""
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == symbol:
                return (r, c)
    raise ValueError(f"Symbol {symbol!r} not found in warehouse map")


def in_bounds(grid, row, col):
    """Check that (row, col) is inside the grid."""
    return 0 <= row < len(grid) and 0 <= col < len(grid[0])


def is_free(grid, row, col):
    """Check that (row, col) is not an obstacle."""
    return grid[row][col] != "#"


class WarehouseAgent:
    """
    A goal-based agent for the warehouse navigation problem.

    The agent maintains:
      - a model of the environment (the grid),
      - its current state (start position),
      - its goal (goal position),
    and uses a search procedure (BFS) as its decision-making component
    to select the sequence of actions that will achieve the goal.
    """

    def __init__(self, grid):
        self.grid = grid
        self.state = find_symbol(grid, "S")   # current state
        self.goal = find_symbol(grid, "G")    # goal state

    def search(self):
        """
        Breadth-First Search over grid cells.

        Returns
        -------
        path : list of (row, col) tuples from start to goal inclusive,
               or None if no path exists.
        actions : list of action names ("Up"/"Down"/"Left"/"Right")
                  corresponding to each step of the path, or None.
        """
        start, goal = self.state, self.goal
        frontier = deque([start])
        came_from = {start: None}          # state -> predecessor state
        action_taken = {start: None}       # state -> action that led here

        while frontier:
            current = frontier.popleft()

            if current == goal:
                return self._reconstruct(came_from, action_taken, goal)

            for action, (dr, dc) in ACTIONS.items():
                nr, nc = current[0] + dr, current[1] + dc
                neighbour = (nr, nc)

                if not in_bounds(self.grid, nr, nc):
                    continue
                if not is_free(self.grid, nr, nc):
                    continue
                if neighbour in came_from:
                    continue  # already visited/queued

                came_from[neighbour] = current
                action_taken[neighbour] = action
                frontier.append(neighbour)

        return None, None  # goal is unreachable

    @staticmethod
    def _reconstruct(came_from, action_taken, goal):
        """Walk backwards from the goal to the start to build the path."""
        path = []
        actions = []
        node = goal
        while node is not None:
            path.append(node)
            if action_taken[node] is not None:
                actions.append(action_taken[node])
            node = came_from[node]
        path.reverse()
        actions.reverse()
        return path, actions

    def run(self):
        """Plan a path and report the result to the user."""
        print("Warehouse map:")
        for row in self.grid:
            print(" ", row)
        print(f"\nStart state S = {self.state}")
        print(f"Goal state  G = {self.goal}\n")

        path, actions = self.search()

        if path is None:
            print("No path exists between S and G: the goal is unreachable.")
            return

        print(f"Path found! Length = {len(path)} cells, "
              f"{len(actions)} moves.\n")
        print("Sequence of grid positions (row, col):")
        print("  " + " -> ".join(str(p) for p in path))
        print("\nSequence of actions:")
        print("  " + ", ".join(actions))


if __name__ == "__main__":
    agent = WarehouseAgent(WAREHOUSE_MAP)
    agent.run()
