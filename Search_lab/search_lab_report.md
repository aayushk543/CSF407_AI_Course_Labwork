# Laboratory Report – Search and A*
## Using an LLM as an Engineering Assistant

**Course:** CSF407 / Artificial Intelligence Laboratory  
**Topic:** Problem Formulation, A* Heuristic Search, Algorithm Validation, and Comparative Evaluation  
**Date:** September 30, 2026  
**Author:** AI Laboratory Student  

---

## 1. Executive Summary

This laboratory exercise investigates the formal formulation, algorithmic implementation, empirical testing, and critical evaluation of **Heuristic Search ($A^*$)** for an autonomous warehouse vehicle navigating obstacles to reach a dispatch area. 

Using Large Language Models (LLMs) as collaborative software engineering assistants under rigorous human supervision, we:
1. Formulate the warehouse navigation problem under the formal 6-tuple $\mathcal{P} = (S, A, T, s_0, G, c)$.
2. Implement $A^*$ with Manhattan distance heuristic ($f(n) = g(n) + h(n)$) using priority queue min-heaps.
3. Validate the implementation across a comprehensive suite of edge cases (original maze, trivial 1-step, disconnected unreachable goal, alternative competing paths).
4. Map core search concepts directly to code elements and verify closed-set cycle-pruning mechanics.
5. Contrast informed $A^*$ search against blind Breadth-First Search (BFS), explaining topology-dependent pruning.
6. Empirically investigate heuristic admissibility across zero ($h=0$), Euclidean, Manhattan, and inadmissible ($2 \times \text{Manhattan}$) formulations.
7. Critically evaluate the strengths and risks of LLM-assisted software engineering.

---

## 2. Task 0: Understand the Search Problem

### 2.1 Formal Search Formulation
A search problem is formally defined by the tuple $\mathcal{P} = (S, A, T, s_0, G, c)$:

| Component | Specification | Description |
|:---|:---|:---|
| **State Space $S$** | $(r, c) \in \{0, \dots, 8\} \times \{0, \dots, 16\}$ | Discrete 2D coordinates of traversable free-space cells (`.`, `S`, `G`). |
| **Actions $A$** | $\{\text{Up}, \text{Down}, \text{Left}, \text{Right}\}$ | Movement vectors: $(-1, 0), (+1, 0), (0, -1), (0, 1)$. |
| **Transition $T$** | $T((r, c), a) = (r + \Delta r_a, c + \Delta c_a)$ | Deterministic movement to an adjacent cell if in-bounds and not an obstacle (`#`). |
| **Initial State $s_0$** | $(1, 1)$ | Start coordinate labeled `'S'`. |
| **Goal States $G$** | $\{(7, 15)\}$ | Target delivery coordinate labeled `'G'`. |
| **Cost Function $c$** | $c(s, a, s') = 1$ | Uniform cost of 1 for every movement step. |

### 2.2 Conceptual Questions
- **(a) What information is necessary to specify a state?**  
  The 2D coordinate tuple $(r, c)$ is necessary and sufficient. Because the vehicle has no orientation, momentum, or payload capacity constraints, its spatial coordinates completely determine all valid future transitions and costs.
- **(b) What makes an action invalid?**  
  An action is invalid if taking it would result in moving outside the grid boundaries ($r' < 0, r' \ge \text{Rows}, c' < 0, c' \ge \text{Cols}$) or colliding with an obstacle shelf (`Grid[r'][c'] == '#'`).
- **(c) Is this a deterministic search problem?**  
  Yes. Every action $a \in A$ transitions the agent from state $s$ to state $s'$ with probability $1.0$ without sensor noise or action uncertainty.
- **(d) What would constitute a solution?**  
  An ordered sequence of actions $\langle a_1, a_2, \dots, a_k \rangle$ that transitions the agent from $s_0$ through valid non-obstacle states to some $s_k \in G$. An *optimal* solution minimizes cumulative cost $\sum_{i=1}^k c(s_{i-1}, a_i, s_i) = k$.

---

## 3. Task 1: Plan the Agent

### 3.1 Architectural Design Specification
1. **State Representation:** Coordinate tuple `(row, col)` of integers.
2. **Warehouse Representation:** A list of strings representing the 2D grid matrix.
3. **Valid Action Determination:** Bounds check `0 <= r < len(grid)` and `0 <= c < len(grid[0])`, plus collision check `grid[r][c] != '#'`.
4. **Goal Recognition:** Exact match `current_state == goal_state`.
5. **Frontier Data Structure:** A min-heap priority queue storing tuples `(f_score, tie_breaker_counter, state)`, prioritizing states with lowest $f(n) = g(n) + h(n)$.
6. **Path Reconstruction:** A predecessor dictionary `came_from[state] = parent_state` and `action_taken[state] = action`, back-tracked from $G$ to $S$.
7. **Telemetry Reporting:** Boolean solution flag, ordered coordinate path, action sequence, total path length (moves), and total number of states expanded.

---

## 4. Task 2: Prompt Engineering & A* Implementation

### 4.1 Exact Prompt Provided to LLM
```text
I am implementing a simple goal-based search agent in Python.
The environment is a grid represented by an ASCII map. The agent starts at S
and must reach G. The symbols # represent obstacles and . represents free cells.
The agent can move up, down, left, or right, and every movement has cost 1.
Implement A* search.
Use Manhattan distance as the heuristic:
h(n) = |x - xG| + |y - yG|.

The program should:
- represent grid positions as states;
- maintain an appropriate frontier;
- calculate g(n), h(n) and f(n);
- avoid repeatedly expanding the same state;
- reconstruct the path when the goal is reached;
- report the path and its length;
- report the number of states expanded.

Keep the implementation simple and explain the main components of the code.
```

---

## 5. Task 3: Systematic Testing of the Generated Program

We subjected the implementation to four deliberate verification test cases:

| Test Case | Scenario | Expected Behavior | Observed Result | Status |
|:---|:---|:---|:---|:---:|
| **Test 1: Original Warehouse** | Full 9 $\times$ 17 map with serpentine obstacle walls | Finds collision-free optimal path | Found 40-step path, expanded 64 states | **PASS** |
| **Test 2: Trivial Case** | Goal immediately adjacent: `#####` / `#SG##` / `#####` | Immediately reaches goal in 1 step | Found 1-step path, expanded 2 states | **PASS** |
| **Test 3: No Solution Case** | Goal completely walled off by obstacles | Terminates gracefully reporting failure (no infinite loop) | Correctly returned `found: False`, expanded 9 states | **PASS** |
| **Test 4: Alternative Paths** | Two routes: shorter direct path (length 6) vs longer detour (length 10) | Must select the shorter optimal route | Selected direct 6-step path, expanded 7 states | **PASS** |

### Test 1 Trajectory Output:
- **Path Length:** 40 moves (41 cells).
- **Coordinate Trajectory:**
  $$(1, 1) \to (1, 2) \to (1, 3) \to (1, 4) \to (1, 5) \to (2, 5) \to (3, 5) \to (4, 5) \to (5, 5) \to (5, 6) \to (5, 7) \to (5, 8) \to (5, 9) \to (5, 10) \to (5, 11) \to (5, 12) \to (5, 13) \to (4, 13) \to (3, 13) \to (3, 12) \to (3, 11) \to (3, 10) \to (3, 9) \to (3, 8) \to (3, 7) \to (2, 7) \to (1, 7) \to (1, 8) \to (1, 9) \to (1, 10) \to (1, 11) \to (1, 12) \to (1, 13) \to (1, 14) \to (1, 15) \to (2, 15) \to (3, 15) \to (4, 15) \to (5, 15) \to (6, 15) \to (7, 15)$$
- **Actions:** $\text{Right} \times 4 \to \text{Down} \times 4 \to \text{Right} \times 8 \to \text{Up} \times 2 \to \text{Left} \times 6 \to \text{Up} \times 2 \to \text{Right} \times 8 \to \text{Down} \times 6$.

---

## 6. Task 4: Inspect the A* Algorithm

### 6.1 Concept Mapping Table
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

### 6.2 Inspection Questions
- **(a) What data structure is used for the A* frontier?**  
  A min-heap priority queue implemented via Python's standard `heapq` module, storing `(f_score, tie_breaker_counter, state)` tuples.
- **(b) How does the program select the next state to expand?**  
  By popping the minimum element from the min-heap via `heapq.heappop(frontier)`, which selects the state with the lowest $f(n) = g(n) + h(n)$.
- **(c) Where is the heuristic calculated?**  
  In `h_val = heuristic_weight * heuristic_func(neighbor, goal)` before pushing successor states into the frontier.
- **(d) Does the program explicitly calculate $f(n) = g(n) + h(n)$?**  
  Yes. The code computes `tentative_g = g_score[current] + 1`, calculates `h_val`, and sums them: `f_val = tentative_g + h_val`.
- **(e) How does the program prevent unnecessary repeated exploration?**  
  Via two complementary mechanisms:
  1. A `closed_set` (hash set) that skips states that have already been popped and expanded.
  2. A `g_score` dictionary that only pushes a neighbor to the frontier if the newly discovered path offers a strictly lower $g$-cost than previously recorded.

---

## 7. Task 5: Compare A* with Blind Search (BFS)

### 7.1 Experimental Comparison Table
We executed both Breadth-First Search (blind) and $A^*$ Search (informed Manhattan) on the original warehouse map:

| Measure | BFS (Blind) | A* (Informed Manhattan) |
|:---|:---:|:---:|
| **Solution found** | **True** | **True** |
| **Path length** | **40** | **40** |
| **States expanded** | **64** | **64** |

### 7.2 Comparative Analysis
- **(a) Did both algorithms find a solution?** Yes. Both successfully reached the goal.
- **(b) Did they find paths of the same length?** Yes. Both found the mathematically optimal path of length 40.
- **(c) Which algorithm expanded fewer states?** On this specific map, both expanded exactly 64 states.
- **(d) Why did A* expand the same number of states on this map?**  
  The original warehouse map consists of a single narrow, winding serpentine corridor surrounded by impassable shelf walls. The entire map contains exactly 64 reachable traversable cells, and the destination $G$ is positioned at the very end of this one-way channel. Because there are no open alternative corridors or unpromising branches, every reachable cell must be explored by both algorithms.
- **When does A* expand fewer states?**  
  In environments with open spaces and multiple branching paths (such as the warehouse from the Agents lab), $A^*$ expands significantly fewer states (**23 states vs 59 states for BFS, a $>60\%$ reduction**) by using its heuristic to pull search directly toward the goal rather than exploring outward in an undirected radial wavefront.

---

## 8. Task 6: Heuristic Investigations & Admissibility

### 8.1 Why Manhattan Distance is Appropriate
When an agent is restricted to 4-way orthogonal movements (Up, Down, Left, Right) on a grid with unit costs, the shortest distance between $(r_1, c_1)$ and $(r_2, c_2)$ in the absence of obstacles is strictly $|r_1 - r_2| + |c_1 - c_2|$.
Because adding obstacles can only increase the actual path length, the Manhattan distance never overestimates the true remaining cost:
$$h_{\text{Manhattan}}(n) \le h^*(n)$$
Thus, Manhattan distance is strictly **admissible** and **consistent** ($h(n) \le c(n, a, n') + h(n')$), guaranteeing that graph-search $A^*$ returns an optimal path.

### 8.2 Heuristic Experimental Results

| Heuristic Formulation | Admissible? | Found? | Path Length | States Expanded |
|:---|:---:|:---:|:---:|:---:|
| **$h(n) = 0$ (Uniform Cost / Dijkstra)** | **Yes** | True | 40 | 64 |
| **Euclidean Distance** | **Yes** | True | 40 | 64 |
| **Manhattan Distance (Standard)** | **Yes** | True | 40 | 64 |
| **$2 \times \text{Manhattan Distance}$ (Weighted $A^*$)** | **No** (Inadmissible) | True | 40 | 64 |

### 8.3 Analysis of Heuristic Variations
1. **$h(n) = 0$ (Blind/Dijkstra):** When $h(n) = 0$, $f(n) = g(n)$. The search becomes purely cost-driven. It remains admissible and optimal, but expands nodes uniformly in all directions without goal guidance.
2. **Euclidean Distance:** Because the straight-line Euclidean metric $\sqrt{(\Delta r)^2 + (\Delta c)^2} \le |\Delta r| + |\Delta c|$, it is also admissible. However, because it assumes diagonal shortcuts that the 4-way robot cannot execute, it is strictly less informed (dominated by Manhattan distance) on grid graphs.
3. **Overestimating Heuristic ($2 \times h(n)$):** Multiplying by 2 violates admissibility ($h(n) > h^*(n)$). While it aggressively drives the frontier toward the goal—often expanding fewer nodes in open graphs—it sacrifices the mathematical guarantee of optimality and can return a suboptimal path in graphs with multiple routes.

---

## 9. Task 7: Evaluation of the LLM as an Engineering Tool

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

## 10. Final Reflection (Section 6)

### 1. Why is it important to formulate the search problem before writing the search algorithm?
Formulating the search problem establishes the mathematical abstraction $\mathcal{P} = (S, A, T, s_0, G, c)$ independently of implementation details. It clarifies what constitutes a state, defines legal transitions, and specifies the optimality criterion. Without prior formulation, an engineer risks encoding incorrect transition semantics, missing boundary conditions, or implementing an algorithm that solves an unintended objective.

### 2. In what sense is A* an "informed" search algorithm?
$A^*$ is informed because it utilizes problem-specific domain knowledge through a heuristic function $h(n)$ to estimate the cost remaining to the goal. Unlike blind search algorithms (BFS/DFS) that expand states solely based on historical path length or depth, $A^*$ uses $f(n) = g(n) + h(n)$ to prioritize states that appear closest to the objective, orienting search exploration toward the goal.

### 3. Why does the choice of heuristic matter?
The choice of heuristic dictates both the computational efficiency and the solution optimality of $A^*$. An admissible and consistent heuristic guarantees that the first goal popped from the frontier is optimal. An informed heuristic that closely approximates the true remaining cost $h^*(n)$ dramatically prunes the search space, whereas an uninformative heuristic ($h=0$) degrades to exhaustive blind search, and an inadmissible heuristic ($h > h^*$) risks returning suboptimal paths.

### 4. What did the LLM contribute to the engineering process?
The LLM acted as an efficient engineering accelerator, rapidly translating conceptual algorithmic designs into clean, boilerplate Python code. It set up the priority queue mechanics, neighbor transition loops, and coordinate manipulation in seconds, allowing the engineer to focus higher-level effort on validation testing, edge cases, and comparative analysis.

### 5. What could go wrong if an engineer simply accepted LLM-generated code without testing it?
Blindly accepting LLM code risks deploying programs with subtle algorithmic flaws: heuristic inadmissibility, incorrect tie-breaking, failure to detect disconnected goals (infinite loops), or memory leaks from improper closed-set management. In critical autonomous systems (such as warehouse robotics), untested code can lead to collisions, suboptimal routing, or deadlock.
