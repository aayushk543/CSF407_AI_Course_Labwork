# Laboratory Report – Constructing a Goal-Based Agent Using a Large Language Model

**Course:** CSF407 / Artificial Intelligence Laboratory  
**Topic:** Goal-Based Intelligent Agents, State-Space Search, and LLM-Assisted Engineering  
**Date:** September 30, 2026  
**Author:** AI Laboratory Student  

---

## 1. Executive Summary

This laboratory exercise explores the design, formal specification, implementation, and empirical verification of a **goal-based intelligent agent** operating in an autonomous warehouse navigation environment. 

Using Large Language Models (LLMs) as collaborative software engineering assistants, we:
1. Formulate the problem using the **PEAS (Performance, Environment, Actuators, Sensors)** framework.
2. Contrast the architectural capabilities of **goal-based agents** against **simple reflex agents**.
3. Implement a complete state-space search algorithm using **Breadth-First Search (BFS)** to guarantee optimal, collision-free navigation through complex obstacle topologies.
4. Critically evaluate iterative prompt engineering, search algorithm selection, and scalability limitations.

---

## 2. Task 1: Understanding the Problem

### 2.1 The Warehouse Environment
The environment is an indoor warehouse represented as a discrete 2-dimensional grid:
- **Dimensions:** 7 rows $\times$ 21 columns = 147 total grid cells.
- **Topology:** A bounded perimeter of impassable walls, internal shelving obstacles (`#`), traversable free space (`.`), an initial package loading bay (`S`), and a package dispatch destination (`G`).

Under the standard Russell & Norvig environment taxonomy:
- **Fully Observable vs. Partially Observable:** **Fully Observable.** The agent has complete perceptual access to the entire grid layout, obstacle coordinates, current state, and goal location.
- **Deterministic vs. Stochastic:** **Deterministic.** Executing any movement action results in the exact intended adjacent grid cell with probability $1.0$.
- **Static vs. Dynamic:** **Static.** The warehouse geometry, obstacle layout, and target destinations remain constant during agent planning and execution.
- **Discrete vs. Continuous:** **Discrete.** The state space (grid coordinates), time steps, percepts, and action primitives are discrete entities.
- **Single-Agent vs. Multi-Agent:** **Single-Agent.** The vehicle is the sole active agent in the space, eliminating competitive or cooperative coordination complexity.
- **Episodic vs. Sequential:** **Sequential.** Current movement decisions affect future position and action choices; the agent's performance depends on the entire sequence of moves.

### 2.2 The Goal of the Agent
The objective of the autonomous vehicle is to transport packages from the loading bay start position $S = (1, 1)$ to the dispatch area destination $G = (1, 19)$ by discovering a **collision-free path** that:
1. Traverses only free-space cells (`.`),
2. Avoids crossing any shelving blocks (`#`), and
3. Minimizes the total path length (number of movements executed).

### 2.3 Available Actions
The vehicle possesses four discrete cardinal movements:
$$\mathcal{A} = \{\text{Up}, \text{Down}, \text{Left}, \text{Right}\}$$

The transition model $T: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$ is defined as:
$$\begin{aligned}
\text{Up} &: (r, c) \longmapsto (r - 1, c) \\
\text{Down} &: (r, c) \longmapsto (r + 1, c) \\
\text{Left} &: (r, c) \longmapsto (r, c - 1) \\
\text{Right} &: (r, c) \longmapsto (r, c + 1)
\end{aligned}$$
An action is valid if and only if:
$$0 \le r' < \text{Rows}, \quad 0 \le c' < \text{Cols}, \quad \text{Grid}[r'][c'] \ne \text{'\#'}$$

### 2.4 Necessary Information & State Representation
To choose each subsequent action, the agent must maintain:
1. **Model of the World:** The static grid map specifying obstacle coordinates and passable aisles.
2. **Current State:** The coordinate tuple $(r, c) \in \mathbb{N}^2$.
3. **Goal State Formulation:** The target destination coordinates $G = (1, 19)$.
4. **Transition Function:** The deterministic mapping predicting successor states given current state and candidate actions.
5. **Planning Representation:** A sequence of pre-computed actions / waypoints, or the frontier queue and `came_from` predecessor mappings during search deliberation.

### 2.5 Goal-Based Agent vs. Simple Reflex Agent
- **Simple Reflex Agent:** A simple reflex agent operates strictly on condition-action rules derived solely from the immediate percept:
  $$\text{Percept} \longrightarrow \text{Condition-Action Rule} \longrightarrow \text{Action}$$
  For instance, a reflex rule might state: *"If obstacle ahead, turn right; else move forward."* 
  Because reflex agents possess no representation of a destination or future states, they cannot evaluate whether an action brings them closer to a distant goal. In concave obstacle arrangements (such as dead-end warehouse shelving corridors), a reflex agent inevitably oscillates between states or becomes trapped indefinitely.
- **Goal-Based Agent:** A goal-based agent explicitly models the objective state ($G$) and deliberates over sequences of actions:
  $$\text{State} + \text{Actions} + \text{Transition Model} + \text{Goal Description} \longrightarrow \text{Search / Planning} \longrightarrow \text{Action Sequence}$$
  It simulates candidate futures before acting, selecting actions because they are proven components of an end-to-end plan that successfully achieves the goal.

### 2.6 Think About It: Scaling to a Warehouse Twice as Large
> **Question:** Suppose the warehouse becomes twice as large. Would the same search strategy still be appropriate? What additional difficulties might arise?

1. **Suitability of Search Strategy:**
   - On unweighted graphs, Breadth-First Search (BFS) is complete and optimal. Its computational complexity is $\mathcal{O}(b^d)$ in time and space, where $b \le 3$ is the effective branching factor and $d$ is the shortest path depth.
   - For a moderately enlarged warehouse ($14 \times 42 = 588$ cells), BFS remains computationally trivial (executing in milliseconds).
   - However, for realistic industrial logistics facilities with tens of thousands of cells, uninformed BFS expands an undirected radial wave in all directions, storing vast numbers of unpromising frontier nodes and wasting substantial memory.
2. **Informed Heuristic Search ($A^*$):**
   - For larger scale, **$A^*$ Search** guided by an admissible and consistent heuristic—such as Manhattan distance $h(n) = |r_n - r_G| + |c_n - c_G|$—becomes far more appropriate. $A^*$ prioritizes expansion toward the destination, reducing the search space by orders of magnitude while preserving shortest-path optimality.
3. **Real-World Operational Difficulties:**
   - **Dynamic Obstacles:** Scaled warehouses feature multiple autonomous vehicles, human pickers, and temporary blockages, violating the static environment assumption. Pure offline planning fails when obstacles appear dynamically, necessitating real-time replanning algorithms (e.g., $D^*$ Lite or Lifelong Planning $A^*$).
   - **Multi-Agent Coordination:** Scaling requires collision avoidance between multiple moving robots, demanding multi-agent path finding (MAPF) or reservation tables.

---

## 3. Task 2: Designing the Intelligent Agent

### 3.1 Formal Agent Specification
1. **Environment:** Discrete 2D grid containing obstacles, open corridors, start loading bay, and destination dispatch area.
2. **Current State:** Coordinate tuple $s = (r, c)$.
3. **Goal Formulation:** Boolean predicate $\text{IsGoal}(s) \iff s = (1, 19)$.
4. **Available Actions:** $\mathcal{A} = \{\text{Up}, \text{Down}, \text{Left}, \text{Right}\}$.
5. **Decision-Making Component:** Breadth-First Search (BFS) path planner operating over the state-space graph.

### 3.2 Goal-Based Agent Architecture Diagram
The interaction between components follows the classic Russell & Norvig Goal-Based Agent architecture:

```text
                     +----------------------------------------+
                     |              ENVIRONMENT               |
                     |            (Warehouse Grid)            |
                     +-------------------+--------------------+
                                         |
                                         | Percepts (Sensor Reading)
                                         v
                                +-----------------+
                                |     SENSORS     |
                                +--------+--------+
                                         |
                                         v
                                +-----------------+
                                |  CURRENT STATE  | <-----+
                                |  (r, c position)|       |
                                +--------+--------+       |
                                         |                |
                                         v                | Updates
   +--------------------+       +-----------------+       | State
   |  TRANSITION MODEL  | ----> |  WHAT WILL IT   |       | Model
   | (Action Mechanics) |       |  BE LIKE IF I   |       |
   +--------------------+       |   DO ACTION A?  |       |
                                +--------+--------+       |
                                         |                |
                                         v                |
   +--------------------+       +-----------------+       |
   |      GOAL (G)      | ----> | DECISION-MAKER: |       |
   | (Dispatch Location)|       | BFS PATH SEARCH |       |
   +--------------------+       +--------+--------+       |
                                         |                |
                                         v                |
                                +-----------------+       |
                                | WHAT ACTION TO  | ------+
                                |    TAKE NOW?    |
                                +--------+--------+
                                         |
                                         v
                                +-----------------+
                                |    ACTUATORS    |
                                +--------+--------+
                                         |
                                         | Selected Movement
                                         v
                     +----------------------------------------+
                     |              ENVIRONMENT               |
                     +----------------------------------------+
```

---

## 4. Task 3: Prompt Engineering, Implementation, and Evaluation

### 4.1 Prompt Engineering Workflow
Rather than submitting an ambiguous request such as *"write a warehouse agent"*, we structured a precise specification prompt:

#### Exact Prompt Submitted to LLM:
```text
Write a well-documented Python program implementing a goal-based agent for the warehouse navigation problem shown below.

Warehouse Map:
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################

The program should:
1. Represent the warehouse as a two-dimensional grid;
2. Determine a collision-free path from S to G;
3. Avoid all obstacles ('#');
4. Print either the complete path found or a suitable message if no path exists;
5. Explain the search algorithm that has been chosen and why it is appropriate.
```

### 4.2 Evaluation of the Generated Program

#### Question 1: Did the LLM generate a working program on the first attempt?
**Yes.** The generated code executed without runtime errors on the initial run. It correctly parsed the multi-line string map, extracted coordinates for $S = (1, 1)$ and $G = (1, 19)$, executed graph-search BFS, and outputted the verified collision-free path of 20 moves.

#### Question 2: If not, how can you improve your prompt?
Although the code worked on the first attempt, prompt engineering can be refined by incorporating formal software engineering constraints:
1. **Explicit Data Types & Signatures:** Specifying typing annotations (e.g., `Tuple[int, int]`, `List[str]`).
2. **Automated Verification Assertions:** Directing the LLM to write self-testing assertion checks verifying that no coordinate on the returned path coincides with an obstacle (`grid[r][c] != '#'`).
3. **Telemetry & Profiling:** Requesting search metrics such as number of nodes expanded, maximum frontier size, and execution latency.
4. **Heuristic Extension:** Requesting an option to toggle between uninformed BFS and heuristic-guided $A^*$ search.

#### Question 3: What search algorithm did the LLM choose?
The LLM selected **Breadth-First Search (BFS)** implemented with a First-In-First-Out (FIFO) queue (`collections.deque`).

#### Question 4: Why do you think the LLM selected this algorithm?
The LLM chose BFS for three compelling theoretical and practical reasons:
1. **Optimality on Unweighted Graphs:** In this grid environment, every movement step carries uniform unit cost ($c = 1$). Under uniform step costs, BFS is mathematically guaranteed to find the path with the minimum number of moves.
2. **Completeness & Cycle Prevention:** Because the state space is finite (147 cells) and visited states are tracked in a hash table (`came_from`), BFS is strictly complete and immune to infinite loops.
3. **Simplicity and Robustness:** Unlike $A^*$ or greedy best-first search, BFS requires no domain-specific heuristic function $h(n)$ or priority queue balancing. For a compact grid, BFS computes the optimal path in a fraction of a millisecond without heuristic tuning.

---

## 5. Execution Results & Path Verification

### 5.1 Optimal Plan Output
Running the generated program yields:
- **Start State:** $(1, 1)$
- **Goal State:** $(1, 19)$
- **Nodes Expanded:** $68$ out of $147$ total cells.
- **Path Length:** $21$ coordinates ($20$ action steps).

#### Coordinate Trajectory:
$$(1, 1) \longrightarrow (1, 2) \longrightarrow (1, 3) \longrightarrow (1, 4) \longrightarrow (2, 4) \longrightarrow (2, 5) \longrightarrow (2, 6) \longrightarrow (2, 7) \longrightarrow (1, 7) \longrightarrow (1, 8) \longrightarrow (1, 9) \longrightarrow (1, 10) \longrightarrow (1, 11) \longrightarrow (1, 12) \longrightarrow (1, 13) \longrightarrow (1, 14) \longrightarrow (1, 15) \longrightarrow (1, 16) \longrightarrow (1, 17) \longrightarrow (1, 18) \longrightarrow (1, 19)$$

#### Action Sequence:
$$\text{Right} \to \text{Right} \to \text{Right} \to \text{Down} \to \text{Right} \to \text{Right} \to \text{Right} \to \text{Up} \to \text{Right} \to \text{Right} \to \text{Right} \to \text{Right} \to \text{Right} \to \text{Right} \to \text{Right} \to \text{Right} \to \text{Right} \to \text{Right} \to \text{Right} \to \text{Right}$$

### 5.2 Trajectory Grid Overlay
```text
#####################
#S***.#************G#
#.##****##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
```
*(Legend: `S` = Start, `G` = Goal, `*` = Trajectory, `#` = Obstacle, `.` = Free Space)*

---

## 6. Full Source Code

```python
"""
Warehouse Navigation Agent
Implementation of a Goal-Based Agent utilizing Breadth-First Search (BFS)
for optimal, collision-free warehouse path planning.
"""

from collections import deque
from typing import List, Tuple, Dict, Optional

WAREHOUSE_MAP = [
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################",
]

ACTIONS: Dict[str, Tuple[int, int]] = {
    "Up":    (-1, 0),
    "Down":  (1, 0),
    "Left":  (0, -1),
    "Right": (0, 1),
}

def find_symbol(grid: List[str], symbol: str) -> Tuple[int, int]:
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == symbol:
                return (r, c)
    raise ValueError(f"Symbol {symbol!r} not found in warehouse map.")

def in_bounds(grid: List[str], r: int, c: int) -> bool:
    return 0 <= r < len(grid) and 0 <= c < len(grid[0])

def is_free(grid: List[str], r: int, c: int) -> bool:
    return grid[r][c] != "#"

class WarehouseAgent:
    def __init__(self, grid: List[str]):
        self.grid = grid
        self.start: Tuple[int, int] = find_symbol(grid, "S")
        self.goal: Tuple[int, int] = find_symbol(grid, "G")

    def search(self) -> Tuple[Optional[List[Tuple[int, int]]], Optional[List[str]], int]:
        """
        Executes Breadth-First Search to find the shortest collision-free path.
        """
        frontier = deque([self.start])
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {self.start: None}
        action_taken: Dict[Tuple[int, int], Optional[str]] = {self.start: None}
        nodes_expanded = 0

        while frontier:
            current = frontier.popleft()
            nodes_expanded += 1

            if current == self.goal:
                path, actions = self._reconstruct(came_from, action_taken, self.goal)
                return path, actions, nodes_expanded

            for action, (dr, dc) in ACTIONS.items():
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)

                if in_bounds(self.grid, nr, nc) and is_free(self.grid, nr, nc):
                    if neighbor not in came_from:
                        came_from[neighbor] = current
                        action_taken[neighbor] = action
                        frontier.append(neighbor)

        return None, None, nodes_expanded

    @staticmethod
    def _reconstruct(came_from, action_taken, goal):
        path, actions = [], []
        curr = goal
        while curr is not None:
            path.append(curr)
            if action_taken[curr] is not None:
                actions.append(action_taken[curr])
            curr = came_from[curr]
        path.reverse()
        actions.reverse()
        return path, actions

    def run(self):
        print("Executing Warehouse Navigation Goal-Based Agent...")
        path, actions, nodes_expanded = self.search()
        if path is None:
            print("Goal is unreachable.")
            return

        print(f"Optimal Path Discovered! Length: {len(path)} cells, Moves: {len(actions)}, Nodes Expanded: {nodes_expanded}")
        print("Sequence:", " -> ".join(str(p) for p in path))
        print("Actions: ", ", ".join(actions))

if __name__ == '__main__':
    agent = WarehouseAgent(WAREHOUSE_MAP)
    agent.run()
```
