import heapq
import math
from collections import deque

ORIGINAL_WAREHOUSE = [
    "#################",
    "#S....#.........#",
    "#.###.#.#######.#",
    "#...#.#.......#.#",
    "###.#.#######.#.#",
    "#...#.........#.#",
    "#.###########.#.#",
    "#.............#G#",
    "#################",
]

TRIVIAL_WAREHOUSE = [
    "#####",
    "#SG##",
    "#####",
]

NO_SOLUTION_WAREHOUSE = [
    "#######",
    "#S....#",
    "###.###",
    "#...#G#",
    "#######",
]

# Alternative paths: top path (longer: 10 steps), bottom path (shorter: 6 steps)
ALTERNATIVE_PATHS_WAREHOUSE = [
    "#########",
    "#S.....G#",
    "#.#####.#",
    "#.......#",
    "#########",
]

ACTIONS = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}

def find_symbol(grid, symbol):
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == symbol:
                return (r, c)
    raise ValueError(f"Symbol {symbol!r} not found in grid")

def in_bounds(grid, r, c):
    return 0 <= r < len(grid) and 0 <= c < len(grid[0])

def is_free(grid, r, c):
    return grid[r][c] != "#"

def get_neighbors(grid, state):
    r, c = state
    neighbors = []
    for action, (dr, dc) in ACTIONS.items():
        nr, nc = r + dr, c + dc
        if in_bounds(grid, nr, nc) and is_free(grid, nr, nc):
            neighbors.append(((nr, nc), action))
    return neighbors

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# -------------------------------------------------------------
# A* Algorithm Implementation
# -------------------------------------------------------------
def a_star_search(grid, heuristic_func=manhattan_distance, heuristic_weight=1.0):
    start = find_symbol(grid, "S")
    goal = find_symbol(grid, "G")

    # Priority queue: (f_score, tie_breaker, state)
    counter = 0
    h_start = heuristic_weight * heuristic_func(start, goal)
    frontier = [(h_start, counter, start)]
    
    g_score = {start: 0}
    came_from = {start: None}
    action_taken = {start: None}
    closed_set = set()
    states_expanded = 0

    while frontier:
        f, _, current = heapq.heappop(frontier)

        if current in closed_set:
            continue
        closed_set.add(current)
        states_expanded += 1

        if current == goal:
            # Reconstruct path
            path, actions = [], []
            curr = goal
            while curr is not None:
                path.append(curr)
                if action_taken[curr] is not None:
                    actions.append(action_taken[curr])
                curr = came_from[curr]
            path.reverse()
            actions.reverse()
            return {
                "found": True,
                "path": path,
                "actions": actions,
                "path_length": len(actions),
                "states_expanded": states_expanded,
            }

        for neighbor, action in get_neighbors(grid, current):
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                h_val = heuristic_weight * heuristic_func(neighbor, goal)
                f_val = tentative_g + h_val
                counter += 1
                heapq.heappush(frontier, (f_val, counter, neighbor))
                came_from[neighbor] = current
                action_taken[neighbor] = action

    return {
        "found": False,
        "path": None,
        "actions": None,
        "path_length": None,
        "states_expanded": states_expanded,
    }

# -------------------------------------------------------------
# Breadth-First Search (BFS) Implementation
# -------------------------------------------------------------
def bfs_search(grid):
    start = find_symbol(grid, "S")
    goal = find_symbol(grid, "G")

    frontier = deque([start])
    came_from = {start: None}
    action_taken = {start: None}
    visited = {start}
    states_expanded = 0

    while frontier:
        current = frontier.popleft()
        states_expanded += 1

        if current == goal:
            path, actions = [], []
            curr = goal
            while curr is not None:
                path.append(curr)
                if action_taken[curr] is not None:
                    actions.append(action_taken[curr])
                curr = came_from[curr]
            path.reverse()
            actions.reverse()
            return {
                "found": True,
                "path": path,
                "actions": actions,
                "path_length": len(actions),
                "states_expanded": states_expanded,
            }

        for neighbor, action in get_neighbors(grid, current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                action_taken[neighbor] = action
                frontier.append(neighbor)

    return {
        "found": False,
        "path": None,
        "actions": None,
        "path_length": None,
        "states_expanded": states_expanded,
    }

def run_all_experiments():
    print("=" * 70)
    print("TASK 3: SYSTEMATIC TESTING OF A* SEARCH")
    print("=" * 70)
    
    # Test 1: Original Warehouse
    res_orig = a_star_search(ORIGINAL_WAREHOUSE)
    print("\n[Test 1: Original Warehouse]")
    print(f"Solution found:  {res_orig['found']}")
    print(f"Path length:     {res_orig['path_length']}")
    print(f"States expanded: {res_orig['states_expanded']}")
    print("Path coordinates:")
    print(" -> ".join(str(p) for p in res_orig['path']))
    print("Action sequence:")
    print(", ".join(res_orig['actions']))

    # Test 2: Trivial Case
    res_triv = a_star_search(TRIVIAL_WAREHOUSE)
    print("\n[Test 2: Trivial Case (#SG##)]")
    print(f"Solution found:  {res_triv['found']}")
    print(f"Path length:     {res_triv['path_length']}")
    print(f"States expanded: {res_triv['states_expanded']}")
    print(f"Actions:         {res_triv['actions']}")

    # Test 3: No Solution Case
    res_nosol = a_star_search(NO_SOLUTION_WAREHOUSE)
    print("\n[Test 3: No Solution Case (Inaccessible Goal)]")
    print(f"Solution found:  {res_nosol['found']}")
    print(f"States expanded: {res_nosol['states_expanded']}")

    # Test 4: Alternative Paths
    res_alt = a_star_search(ALTERNATIVE_PATHS_WAREHOUSE)
    print("\n[Test 4: Alternative Paths]")
    print(f"Solution found:  {res_alt['found']}")
    print(f"Path length:     {res_alt['path_length']} (Checking if optimal shortest path: {res_alt['path_length'] == 6})")
    print(f"States expanded: {res_alt['states_expanded']}")
    print(f"Actions:         {res_alt['actions']}")

    print("\n" + "=" * 70)
    print("TASK 5: COMPARISON BETWEEN BFS AND A*")
    print("=" * 70)
    bfs_res = bfs_search(ORIGINAL_WAREHOUSE)
    astar_res = a_star_search(ORIGINAL_WAREHOUSE)

    print(f"{'Measure':<20} | {'BFS':<15} | {'A*':<15}")
    print("-" * 56)
    print(f"{'Solution found':<20} | {str(bfs_res['found']):<15} | {str(astar_res['found']):<15}")
    print(f"{'Path length':<20} | {str(bfs_res['path_length']):<15} | {str(astar_res['path_length']):<15}")
    print(f"{'States expanded':<20} | {str(bfs_res['states_expanded']):<15} | {str(astar_res['states_expanded']):<15}")

    print("\n" + "=" * 70)
    print("TASK 6: HEURISTIC INVESTIGATIONS")
    print("=" * 70)
    # 1. h(n) = 0
    res_h0 = a_star_search(ORIGINAL_WAREHOUSE, heuristic_func=lambda p1, p2: 0)
    # 2. Euclidean
    res_euclid = a_star_search(ORIGINAL_WAREHOUSE, heuristic_func=euclidean_distance)
    # 3. Manhattan
    res_manhattan = a_star_search(ORIGINAL_WAREHOUSE, heuristic_func=manhattan_distance)
    # 4. 2 * Manhattan
    res_2manhattan = a_star_search(ORIGINAL_WAREHOUSE, heuristic_func=manhattan_distance, heuristic_weight=2.0)

    heuristics = [
        ("h(n) = 0 (Uniform Cost / Dijkstra)", res_h0),
        ("Euclidean Distance", res_euclid),
        ("Manhattan Distance (Standard)", res_manhattan),
        ("2 * Manhattan Distance (Inadmissible)", res_2manhattan),
    ]

    print(f"{'Heuristic Function':<38} | {'Found?':<8} | {'Path Length':<12} | {'States Expanded':<16}")
    print("-" * 80)
    for name, r in heuristics:
        print(f"{name:<38} | {str(r['found']):<8} | {str(r['path_length']):<12} | {str(r['states_expanded']):<16}")

if __name__ == '__main__':
    run_all_experiments()
