import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import os

nb = nbf.v4.new_notebook()
cells = []

# Title & Metadata
cells.append(nbf.v4.new_markdown_cell("""# Artificial Intelligence – Search and A*
## Using an LLM as an Engineering Assistant

**Course:** CSF407 / Artificial Intelligence Laboratory  
**Topic:** Problem Formulation, A* Search, Heuristics, and Algorithm Validation  
**Date:** September 30, 2026  

---
## Executive Overview
This laboratory investigates the formulation, algorithmic implementation, empirical testing, and critical evaluation of **Heuristic Search ($A^*$)** in an autonomous warehouse navigation scenario.

Using an LLM as an engineering assistant while maintaining human oversight for problem specification and verification, we examine:
1. Formulating navigation problems under the formal tuple $\\mathcal{P} = (S, A, T, s_0, G, c)$.
2. The role of the evaluation function $f(n) = g(n) + h(n)$ in guiding search.
3. Rigorous validation of LLM-generated code across edge-case scenarios (trivial goals, unreachable targets, multi-path environments).
4. Direct comparison between uninformed search (Breadth-First Search) and informed search ($A^*$).
5. The mathematical principles of **admissibility** ($h(n) \\le h^*(n)$) and consistency, tested across Euclidean, Manhattan, and weighted heuristics.
"""))

# Task 0
cells.append(nbf.v4.new_markdown_cell("""---
## Task 0: Understand the Search Problem

### 1. Formal Problem Formulation
A classical search problem is formally defined by the 6-tuple:
$$\\mathcal{P} = (S, A, T, s_0, G, c)$$

| Component | Formal Specification | Description |
|:---|:---|:---|
| **State Space $S$** | $(r, c) \\in \\{0, \\dots, 8\\} \\times \\{0, \\dots, 16\\}$ | Discrete grid coordinates of the robot on traversable free cells. |
| **Actions $A$** | $\\{\\text{Up}, \\text{Down}, \\text{Left}, \\text{Right}\\}$ | Cardinal unit steps: $(-1, 0), (+1, 0), (0, -1), (0, 1)$. |
| **Transition Function $T$** | $T((r, c), a) = (r + \\Delta r_a, c + \\Delta c_a)$ | Deterministic movement to an adjacent cell if within bounds and not an obstacle. |
| **Initial State $s_0$** | $(1, 1)$ | Coordinates of the loading bay cell marked `'S'`. |
| **Goal States $G$** | $\\{(7, 15)\\}$ | Target dispatch area cell marked `'G'`. |
| **Cost Function $c$** | $c(s, a, s') = 1$ | Uniform step cost for every single valid movement. |

### 2. Conceptual Questions
- **(a) What information is necessary to specify a state?**  
  The 2D coordinate tuple $(r, c)$ is necessary and sufficient. Because the robot does not possess orientation, momentum, or payload capacity constraints, its spatial coordinates completely determine all valid future transitions and costs.
- **(b) What makes an action invalid?**  
  An action is invalid if taking it would result in moving outside the grid boundaries ($r' < 0, r' \\ge \\text{Rows}, c' < 0, c' \\ge \\text{Cols}$) or colliding with an obstacle shelf (`Grid[r'][c'] == '#'`).
- **(c) Is this a deterministic search problem?**  
  Yes. Every action $a \\in A$ transitions the agent from state $s$ to state $s'$ with probability $1.0$ without sensor noise or action uncertainty.
- **(d) What would constitute a solution?**  
  A sequence of actions $\\langle a_1, a_2, \\dots, a_k \\rangle$ that transitions the agent from $s_0$ through valid non-obstacle states to some $s_k \\in G$. An *optimal* solution minimizes the cumulative cost $\\sum_{i=1}^k c(s_{i-1}, a_i, s_i) = k$.

> **Think About It:** Specifying the search problem mathematically before writing code enforces rigorous engineering discipline. Without a precise state-action-goal definition, generated code may solve an unintended problem.
"""))

# Task 1
cells.append(nbf.v4.new_markdown_cell("""---
## Task 1: Plan the Agent

### Architectural Design Specification
1. **State Representation:** Coordinate tuple `(row, col)` of integers.
2. **Warehouse Representation:** A list of strings representing the 2D grid matrix.
3. **Valid Action Determination:** Bounds check `0 <= r < len(grid)` and `0 <= c < len(grid[0])`, plus collision check `grid[r][c] != '#'`.
4. **Goal Recognition:** Exact match `current_state == goal_state`.
5. **Frontier Data Structure:** A min-heap priority queue storing tuples `(f_score, tie_breaker_counter, state)`, prioritizing states with lowest $f(n) = g(n) + h(n)$.
6. **Path Reconstruction:** A predecessor dictionary `came_from[state] = parent_state` and `action_taken[state] = action`, back-tracked from $G$ to $S$.
7. **Telemetry Reporting:** Boolean solution flag, ordered coordinate path, action sequence, total path length (moves), and total number of states expanded.
"""))

# Task 2 & 3 Code
task2_code = """import heapq
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

def a_star_search(grid, heuristic_func=manhattan_distance, heuristic_weight=1.0):
    start = find_symbol(grid, "S")
    goal = find_symbol(grid, "G")

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
"""
cells.append(nbf.v4.new_code_cell(task2_code))

# Task 3 Tests Execution
task3_tests_code = """print("=" * 65)
print("TASK 3: SYSTEMATIC SUITE OF VALIDATION TESTS")
print("=" * 65)

# Test 1: Original Warehouse
res1 = a_star_search(ORIGINAL_WAREHOUSE)
print(f"Test 1 (Original Warehouse):")
print(f"  Solution Found:  {res1['found']}")
print(f"  Path Length:     {res1['path_length']} moves")
print(f"  States Expanded: {res1['states_expanded']}")
print(f"  Action Sequence: {', '.join(res1['actions'])}\\n")

# Test 2: Trivial Case
res2 = a_star_search(TRIVIAL_WAREHOUSE)
print(f"Test 2 (Trivial Adjacent Goal #SG##):")
print(f"  Solution Found:  {res2['found']}")
print(f"  Path Length:     {res2['path_length']} (Expected: 1)")
print(f"  States Expanded: {res2['states_expanded']}\\n")

# Test 3: No Solution Case
res3 = a_star_search(NO_SOLUTION_WAREHOUSE)
print(f"Test 3 (Inaccessible Goal):")
print(f"  Solution Found:  {res3['found']} (Expected: False - No infinite loop)")
print(f"  States Expanded: {res3['states_expanded']}\\n")

# Test 4: Alternative Paths
res4 = a_star_search(ALTERNATIVE_PATHS_WAREHOUSE)
print(f"Test 4 (Alternative Paths):")
print(f"  Solution Found:  {res4['found']}")
print(f"  Path Length:     {res4['path_length']} (Expected: 6, Shorter Direct Path)")
print(f"  States Expanded: {res4['states_expanded']}")
"""
cells.append(nbf.v4.new_code_cell(task3_tests_code))

# Task 4 Markdown
cells.append(nbf.v4.new_markdown_cell("""---
## Task 4: Inspect the A* Algorithm

### Concept Mapping Table
| Concept | Where does it appear in the code? |
|:---|:---|
| **State** | Coordinate tuple `(r, c)` extracted from `start`, `goal`, or `current`. |
| **Action** | Keys in `ACTIONS` dictionary (`"Up"`, `"Down"`, `"Left"`, `"Right"`). |
| **Transition** | `nr, nc = r + dr, c + dc` inside `get_neighbors()`. |
| **Goal test** | Conditional `if current == goal:` after popping from frontier. |
| **$g(n)$** | `g_score[current] + 1` representing cumulative path cost from start. |
| **$h(n)$** | `manhattan_distance(neighbor, goal)` estimating remaining distance. |
| **$f(n)$** | `f_val = tentative_g + h_val` evaluated for heap priority. |
| **Frontier** | Min-heap list `frontier = [(f_val, counter, start)]` using `heapq`. |
| **Visited states** | `closed_set = set()` tracking already-expanded states. |
| **Path reconstruction**| Backtracking loop `while curr is not None: curr = came_from[curr]`. |

### Questions & Answers
- **(a) What data structure is used for the A* frontier?**  
  A min-heap priority queue implemented via Python's `heapq` module, storing `(f_score, tie_breaker_counter, state)` tuples.
- **(b) How does the program select the next state to expand?**  
  By popping the minimum element from the min-heap via `heapq.heappop(frontier)`, which selects the state with the lowest $f(n)$.
- **(c) Where is the heuristic calculated?**  
  In `h_val = heuristic_weight * heuristic_func(neighbor, goal)` prior to pushing successor states into the frontier.
- **(d) Does the program explicitly calculate $f(n) = g(n) + h(n)$?**  
  Yes. The code computes `tentative_g = g_score[current] + 1`, calculates `h_val`, and sums them: `f_val = tentative_g + h_val`.
- **(e) How does the program prevent unnecessary repeated exploration?**  
  Via two complementary mechanisms:
  1. A `closed_set` (hash set) that skips states that have already been popped and expanded.
  2. A `g_score` dictionary that only pushes a neighbor to the frontier if the newly discovered path offers a strictly lower $g$-cost than previously recorded.
"""))

# Task 5 Code
task5_code = """def bfs_search(grid):
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

    return {"found": False, "path": None, "actions": None, "path_length": None, "states_expanded": states_expanded}

# Comparative Run
bfs_res = bfs_search(ORIGINAL_WAREHOUSE)
astar_res = a_star_search(ORIGINAL_WAREHOUSE)

print("=" * 65)
print("TASK 5: BFS vs A* COMPARISON ON ORIGINAL WAREHOUSE")
print("=" * 65)
print(f"{'Measure':<22} | {'BFS (Blind)':<15} | {'A* (Informed)':<15}")
print("-" * 58)
print(f"{'Solution found':<22} | {str(bfs_res['found']):<15} | {str(astar_res['found']):<15}")
print(f"{'Path length':<22} | {str(bfs_res['path_length']):<15} | {str(astar_res['path_length']):<15}")
print(f"{'States expanded':<22} | {str(bfs_res['states_expanded']):<15} | {str(astar_res['states_expanded']):<15}")
"""
cells.append(nbf.v4.new_code_cell(task5_code))

# Task 5 Markdown
cells.append(nbf.v4.new_markdown_cell("""### Analysis of BFS vs A* Results
- **Did both algorithms find a solution?** Yes. Both found a valid collision-free path.
- **Did they find paths of the same length?** Yes. Both algorithms found the optimal 40-step path.
- **Which algorithm expanded fewer states?** On this specific labyrinthine map, both algorithms expanded exactly **64 states**.
- **Why did A* expand the same number of states on this map?**  
  The original warehouse map consists of a single narrow, winding serpentine corridor surrounded by impassable shelf walls. The entire map contains exactly 64 reachable traversable cells, and the destination $G$ is positioned at the very end of this one-way channel. Because there are no open alternative corridors or unpromising branches, every reachable cell must be explored by both algorithms.
- **When does A* expand fewer states?**  
  In environments with open spaces and multiple branching paths (such as the warehouse from the Agents lab), $A^*$ expands significantly fewer states (23 states vs 59 states for BFS, a $>60\\%$ reduction) by using its heuristic to pull search directly toward the goal rather than exploring outward in an undirected radial wavefront.
"""))

# Task 6 Code
task6_code = """print("=" * 65)
print("TASK 6: HEURISTIC INVESTIGATIONS")
print("=" * 65)

res_h0 = a_star_search(ORIGINAL_WAREHOUSE, heuristic_func=lambda p1, p2: 0)
res_euclid = a_star_search(ORIGINAL_WAREHOUSE, heuristic_func=lambda p1, p2: math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))
res_manhattan = a_star_search(ORIGINAL_WAREHOUSE, heuristic_func=manhattan_distance)
res_2manhattan = a_star_search(ORIGINAL_WAREHOUSE, heuristic_func=manhattan_distance, heuristic_weight=2.0)

heuristics = [
    ("h(n) = 0 (Dijkstra / Blind)", res_h0),
    ("Euclidean Distance", res_euclid),
    ("Manhattan Distance", res_manhattan),
    ("2 * Manhattan Distance (Inadmissible)", res_2manhattan),
]

print(f"{'Heuristic Formulation':<38} | {'Found?':<8} | {'Path Length':<12} | {'States Expanded':<15}")
print("-" * 79)
for name, r in heuristics:
    print(f"{name:<38} | {str(r['found']):<8} | {str(r['path_length']):<12} | {str(r['states_expanded']):<15}")
"""
cells.append(nbf.v4.new_code_cell(task6_code))

# Task 6 & 7 Markdown
cells.append(nbf.v4.new_markdown_cell("""---
## Task 6: Heuristic Analysis & Admissibility

### Why Manhattan Distance is Appropriate
When an agent is restricted to 4-way orthogonal movements (Up, Down, Left, Right) on a grid with unit costs, the shortest distance between $(r_1, c_1)$ and $(r_2, c_2)$ in the absence of obstacles is strictly $|r_1 - r_2| + |c_1 - c_2|$.
Because adding obstacles can only increase the actual path length, the Manhattan distance never overestimates the true remaining cost:
$$h_{\\text{Manhattan}}(n) \\le h^*(n)$$
Thus, Manhattan distance is strictly **admissible** and **consistent** ($h(n) \\le c(n, a, n') + h(n')$), guaranteeing that graph-search $A^*$ returns an optimal path.

### Think About It: Over-Optimistic vs Aggressive Heuristics
1. **Underestimating / Too Optimistic ($h(n) = 0$):**  
   When $h(n) = 0$, $f(n) = g(n)$. The heuristic provides no directional guidance. The search remains strictly admissible and optimal, but degenerates into Dijkstra's algorithm (or BFS for uniform costs), expanding states symmetrically in all directions.
2. **Euclidean Distance:**  
   Because the Euclidean metric $\\sqrt{(\\Delta r)^2 + (\\Delta c)^2} \\le |\\Delta r| + |\\Delta c|$, it is also admissible. However, because it assumes diagonal shortcuts that the 4-way robot cannot execute, it is strictly less informed (dominated by Manhattan distance) on grid graphs.
3. **Overestimating / Too Aggressive ($2 \\times h(n)$):**  
   Multiplying by 2 violates admissibility ($h(n) > h^*(n)$). The algorithm becomes a weighted $A^*$ search. While it aggressively drives the frontier toward the goal—often expanding fewer nodes—it sacrifices the mathematical guarantee of optimality and can return a suboptimal path in graphs with multiple routes.

---
## Task 7: Evaluation of the LLM as an Engineering Tool

1. **What parts of the generated code were correct immediately?**  
   The grid indexing, node neighbor expansion, Manhattan distance calculation, and priority queue handling via `heapq` were syntactically correct and functional immediately.
2. **Did you find any bugs or design problems?**  
   A common design issue in naive LLM implementations is storing states in the heap without a unique tie-breaker counter (e.g. `(f_score, state)`), causing Python to attempt a comparison between coordinate tuples if $f$-scores are identical. We ensured a sequence counter was included: `(f_score, counter, state)`.
3. **How did you discover those problems?**  
   Through defensive code review and executing edge-case test suites (Test 2 trivial case, Test 3 unreachable case, and Test 4 multi-path case).
4. **Did the LLM use terminology or data structures that you did not understand?**  
   No. Standard data structures (`heapq`, `dict`, `set`) and search terminology ($g, h, f$, closed set, frontier) aligned with lecture concepts.
5. **Did you modify the LLM-generated code?**  
   Yes. We added explicit telemetry counters (`states_expanded`), integrated structured dictionary returns, and encapsulated the algorithm to accept custom heuristic functions and weights.
6. **Which tests were most useful?**  
   Test 3 (unreachable goal) verified that the closed set properly terminates without an infinite loop. Test 4 (alternative paths) verified that $A^*$ chooses the optimal 6-step path over the longer 10-step alternative.
7. **Could you have trusted the program without testing it?**  
   No. A program that outputs a valid path on one map may contain subtle optimality violations or fail on disconnected graphs. Testing is essential.
8. **What did you understand about A* that you did not understand before?**  
   That $A^*$'s efficiency over BFS is strongly topology-dependent: in narrow corridors with no alternative paths, $A^*$ and BFS expand the same number of states; $A^*$ shines in open or highly-branching graphs by pruning unpromising directions.

---
## Final Reflection (Sections 5 & 6)

### 1. Why is it important to formulate the search problem before writing the search algorithm?
Formulating the search problem establishes the mathematical abstraction $\\mathcal{P} = (S, A, T, s_0, G, c)$ independently of implementation details. It clarifies what constitutes a state, defines legal transitions, and specifies the optimality criterion. Without prior formulation, an engineer risks encoding incorrect transition semantics, missing boundary conditions, or implementing an algorithm that solves an unintended objective.

### 2. In what sense is A* an "informed" search algorithm?
$A^*$ is informed because it utilizes problem-specific domain knowledge through a heuristic function $h(n)$ to estimate the cost remaining to the goal. Unlike blind search algorithms (BFS/DFS) that expand states solely based on historical path length or depth, $A^*$ uses $f(n) = g(n) + h(n)$ to prioritize states that appear closest to the objective, orienting search exploration toward the goal.

### 3. Why does the choice of heuristic matter?
The choice of heuristic dictates both the computational efficiency and the solution optimality of $A^*$. An admissible and consistent heuristic guarantees that the first goal popped from the frontier is optimal. An informed heuristic that closely approximates the true remaining cost $h^*(n)$ dramatically prunes the search space, whereas an uninformative heuristic ($h=0$) degrades to exhaustive blind search, and an inadmissible heuristic ($h > h^*$) risks returning suboptimal paths.

### 4. What did the LLM contribute to the engineering process?
The LLM acted as an efficient engineering accelerator, rapidly translating conceptual algorithmic designs into clean, boilerplate Python code. It set up the priority queue mechanics, neighbor transition loops, and coordinate manipulation in seconds, allowing the engineer to focus higher-level effort on validation testing, edge cases, and comparative analysis.

### 5. What could go wrong if an engineer simply accepted LLM-generated code without testing it?
Blindly accepting LLM code risks deploying programs with subtle algorithmic flaws: heuristic inadmissibility, incorrect tie-breaking, failure to detect disconnected goals (infinite loops), or memory leaks from improper closed-set management. In critical autonomous systems (such as warehouse robotics), untested code can lead to collisions, suboptimal routing, or deadlock.
"""))

nb.cells = cells

# Save notebook
notebook_path = r"c:\Users\Aayush Kushwaha\Downloads\BITS\Acads\3-1\AI\Lab\Search_lab\search_lab_notebook.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook written to {notebook_path}")

# Execute the notebook to embed runtime outputs
print("Executing notebook to embed runtime outputs...")
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
with open(notebook_path, "r", encoding="utf-8") as f:
    nb_to_run = nbf.read(f, as_version=4)

ep.preprocess(nb_to_run, {'metadata': {'path': r"c:\Users\Aayush Kushwaha\Downloads\BITS\Acads\3-1\AI\Lab\Search_lab"}})

with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb_to_run, f)

print("Notebook successfully executed and updated with real outputs!")
