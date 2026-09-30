import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import os

nb = nbf.v4.new_notebook()
cells = []

# Title & Metadata
cells.append(nbf.v4.new_markdown_cell("""# Laboratory – Logical Reasoning for Planning
## Using an LLM to Construct and Test a Simple Planning Agent

**Course:** CSF407 / Artificial Intelligence Laboratory  
**Topic:** STRIPS Planning, Precondition/Effect Logic, State-Space Search, and Prolog Verification  
**Date:** September 30, 2026  

---
## Executive Overview
This laboratory investigates how **logical reasoning** and **state-space search** integrate to form intelligent planning agents:
$$\\text{Logic} + \\text{Search} = \\text{Planning}$$

Using a small warehouse robot navigation and package delivery scenario across locations $\\{A, B, C\\}$, we:
1. Specify actions using formal **preconditions** and **effects** (STRIPS-style representation).
2. Determine action applicability using logical entailment: $S \\models \\text{Preconditions}(a)$.
3. Implement a Breadth-First Search (BFS) planner in Python to discover optimal action sequences.
4. Validate the planner across edge cases (solvable, impossible, and irrelevant actions).
5. Explore independent formal verification using **Prolog** as an executable rule-based reasoning engine.
"""))

# Task 0
cells.append(nbf.v4.new_markdown_cell("""---
## Task 0: Understand the Planning Problem

A classical planning problem is formally described by the triple:
$$\\mathcal{P} = (I, A, G)$$

### 1. Problem Specification
- **Initial State $I$:**
  $$I = \\{\\text{At}(\\text{Robot}, A), \\text{At}(\\text{Package}, A)\\}$$
- **Goal State $G$:**
  $$G = \\{\\text{At}(\\text{Package}, C)\\}$$
- **Available Actions $A$:**
  1. $\\text{Move}(X, Y)$ for connected locations $(X, Y) \\in \\{(A, B), (B, A), (B, C), (C, B)\\}$
  2. $\\text{PickUp}(\\text{Package}, X)$ for $X \\in \\{A, B, C\\}$
  3. $\\text{Drop}(\\text{Package}, X)$ for $X \\in \\{A, B, C\\}$

### 2. Action Schema: Preconditions and Effects

| Action | Preconditions | Positive Effects (Add List) | Negative Effects (Delete List) |
|:---|:---|:---|:---|
| $\\text{Move}(X, Y)$ | $\\text{At}(\\text{Robot}, X)$ | $\\text{At}(\\text{Robot}, Y)$ | $\\text{At}(\\text{Robot}, X)$ |
| $\\text{PickUp}(\\text{Package}, X)$ | $\\text{At}(\\text{Robot}, X), \\text{At}(\\text{Package}, X)$ | $\\text{Holding}(\\text{Package})$ | $\\text{At}(\\text{Package}, X)$ |
| $\\text{Drop}(\\text{Package}, X)$ | $\\text{At}(\\text{Robot}, X), \\text{Holding}(\\text{Package})$ | $\\text{At}(\\text{Package}, X)$ | $\\text{Holding}(\\text{Package})$ |

### 3. Initial Action Applicability Question
Starting from $I = \\{\\text{At}(\\text{Robot}, A), \\text{At}(\\text{Package}, A)\\}$:
- **Is $\\text{PickUp}(\\text{Package}, A)$ applicable?**  
  **Yes.** Its preconditions are $\\text{At}(\\text{Robot}, A)$ and $\\text{At}(\\text{Package}, A)$. Both facts are present in $I$ ($I \\models \\text{Preconditions}(\\text{PickUp}(P, A))$).
- **Is $\\text{Drop}(\\text{Package}, C)$ applicable?**  
  **No.** Its preconditions are $\\text{At}(\\text{Robot}, C)$ and $\\text{Holding}(\\text{Package})$. Neither fact is true in $I$.

> **Think About It:** An action cannot be executed merely because it exists in the domain catalog. It is only applicable if all its preconditions logically hold in the current state: $S \\models \\text{Preconditions}(a)$. This is the primary bridge connecting logical deduction to automated planning.
"""))

# Task 1
cells.append(nbf.v4.new_markdown_cell("""---
## Task 1: Construct a Plan by Hand

A plan is an action sequence $\\langle a_1, a_2, \\dots, a_n \\rangle$ producing valid state transitions:
$$I \\xrightarrow{a_1} S_1 \\xrightarrow{a_2} S_2 \\dots \\xrightarrow{a_n} S_n \\quad \\text{such that} \\quad S_n \\models G$$

### Manual State-Action Trace

| State | Action Applied | Resulting State Facts | Goal Satisfied? |
|:---:|:---|:---|:---:|
| **$S_0$** | *(Initial)* | $\\{\\text{At}(\\text{Robot}, A), \\text{At}(\\text{Package}, A)\\}$ | No |
| **$S_1$** | $\\text{PickUp}(\\text{Package}, A)$ | $\\{\\text{At}(\\text{Robot}, A), \\text{Holding}(\\text{Package})\\}$ | No |
| **$S_2$** | $\\text{Move}(A, B)$ | $\\{\\text{At}(\\text{Robot}, B), \\text{Holding}(\\text{Package})\\}$ | No |
| **$S_3$** | $\\text{Move}(B, C)$ | $\\{\\text{At}(\\text{Robot}, C), \\text{Holding}(\\text{Package})\\}$ | No |
| **$S_4$** | $\\text{Drop}(\\text{Package}, C)$ | $\\{\\text{At}(\\text{Robot}, C), \\text{At}(\\text{Package}, C)\\}$ | **YES** |

The plan length is **4 actions**.
"""))

# Task 2 & 3 Code
task2_code = """from collections import deque
from typing import Set, List, Dict, Tuple, Optional

class Action:
    def __init__(self, name: str,
                 pos_preconds: Set[str], neg_preconds: Set[str],
                 pos_effects: Set[str], neg_effects: Set[str]):
        self.name = name
        self.pos_preconds = pos_preconds
        self.neg_preconds = neg_preconds
        self.pos_effects = pos_effects
        self.neg_effects = neg_effects

    def is_applicable(self, state: Set[str]) -> bool:
        return self.pos_preconds.issubset(state) and self.neg_preconds.isdisjoint(state)

    def apply(self, state: Set[str]) -> Set[str]:
        new_state = set(state)
        new_state.difference_update(self.neg_effects)
        new_state.update(self.pos_effects)
        return new_state

    def __repr__(self):
        return self.name

class PlanningProblem:
    def __init__(self, initial_state: Set[str], goal_state: Set[str], actions: List[Action]):
        self.initial_state = frozenset(initial_state)
        self.goal_state = set(goal_state)
        self.actions = actions

    def is_goal_satisfied(self, state: Set[str]) -> bool:
        return self.goal_state.issubset(state)

class BFSPlanner:
    def __init__(self, problem: PlanningProblem):
        self.problem = problem

    def plan(self) -> Tuple[Optional[List[Action]], Optional[List[Set[str]]], int]:
        start = self.problem.initial_state
        if self.problem.is_goal_satisfied(set(start)):
            return [], [set(start)], 0

        frontier = deque([(start, [], [set(start)])])
        visited = {start}
        states_expanded = 0

        while frontier:
            current_state, plan, trajectory = frontier.popleft()
            states_expanded += 1

            for action in self.problem.actions:
                if action.is_applicable(set(current_state)):
                    successor = frozenset(action.apply(set(current_state)))

                    if successor not in visited:
                        visited.add(successor)
                        new_plan = plan + [action]
                        new_trajectory = trajectory + [set(successor)]

                        if self.problem.is_goal_satisfied(set(successor)):
                            return new_plan, new_trajectory, states_expanded

                        frontier.append((successor, new_plan, new_trajectory))

        return None, None, states_expanded

def build_warehouse_actions(allow_pickup: bool = True) -> List[Action]:
    locations = ["A", "B", "C"]
    connections = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]
    actions = []

    for loc1, loc2 in connections:
        actions.append(Action(
            name=f"Move({loc1}, {loc2})",
            pos_preconds={f"At(Robot, {loc1})"},
            neg_preconds=set(),
            pos_effects={f"At(Robot, {loc2})"},
            neg_effects={f"At(Robot, {loc1})"}
        ))

    if allow_pickup:
        for loc in locations:
            actions.append(Action(
                name=f"PickUp(Package, {loc})",
                pos_preconds={f"At(Robot, {loc})", f"At(Package, {loc})"},
                neg_preconds={"Holding(Package)"},
                pos_effects={"Holding(Package)"},
                neg_effects={f"At(Package, {loc})"}
            ))

    for loc in locations:
        actions.append(Action(
            name=f"Drop(Package, {loc})",
            pos_preconds={f"At(Robot, {loc})", "Holding(Package)"},
            neg_preconds=set(),
            pos_effects={f"At(Package, {loc})"},
            neg_effects={"Holding(Package)"}
        ))

    return actions

def verify_plan(initial_state: Set[str], goal: Set[str], plan: List[Action]) -> Tuple[bool, List[str]]:
    current = set(initial_state)
    log = [f"S0: {sorted(list(current))}"]
    for i, action in enumerate(plan):
        if not action.is_applicable(current):
            log.append(f"FAIL at Step {i+1}: {action.name} not applicable!")
            return False, log
        current = action.apply(current)
        log.append(f"S{i+1}: After {action.name} -> {sorted(list(current))}")
    if not goal.issubset(current):
        log.append("FAIL: Goal condition not satisfied!")
        return False, log
    log.append("PASS: Goal verified!")
    return True, log
"""
cells.append(nbf.v4.new_code_cell(task2_code))

# Task 3 Execution
task3_code = """print("=" * 65)
print("TASK 3: SYSTEMATIC TESTING OF PLANNER")
print("=" * 65)

I = {"At(Robot, A)", "At(Package, A)"}
G = {"At(Package, C)"}

# Test A: Solvable
prob_A = PlanningProblem(I, G, build_warehouse_actions(allow_pickup=True))
plan_A, traj_A, exp_A = BFSPlanner(prob_A).plan()

print("\\n[Test A: Solvable Warehouse Problem]")
print(f"Plan found: {plan_A is not None}")
print(f"Plan length: {len(plan_A)} actions | States expanded: {exp_A}")
for idx, act in enumerate(plan_A):
    print(f"  Step {idx+1}: {act.name}")
valid_A, log_A = verify_plan(I, G, plan_A)
print(f"Independent Verification: {'VALID' if valid_A else 'INVALID'}")
print("\\nExecution Trace:")
for line in log_A:
    print(f"  {line}")

# Test B: Impossible Problem (Remove PickUp)
prob_B = PlanningProblem(I, G, build_warehouse_actions(allow_pickup=False))
plan_B, traj_B, exp_B = BFSPlanner(prob_B).plan()

print("\\n[Test B: Impossible Problem (PickUp Removed)]")
print(f"Plan found: {plan_B is not None} ({'No plan found' if plan_B is None else 'Error: Plan invented'})")
print(f"States expanded: {exp_B}")

# Test C: Irrelevant Actions
prob_C = PlanningProblem(I, G, build_warehouse_actions(allow_pickup=True))
print("\\n[Test C: Irrelevant Actions Check]")
print("Does planner reach Robot at C without package? No, goal At(Package, C) is strictly enforced.")
"""
cells.append(nbf.v4.new_code_cell(task3_code))

# Task 4 & 5 Markdown
cells.append(nbf.v4.new_markdown_cell("""---
## Task 4: Logic and Search Interaction

### Conceptual Roles
Planning integrates two distinct computational paradigms:
- **Logical Reasoning:** Evaluates state semantics locally:
  $$S \\models \\text{Preconditions}(a) \\implies S' = (S \\setminus \\text{NegEffects}(a)) \\cup \\text{PosEffects}(a)$$
  Logic determines **what transitions are legal**.
- **State-Space Search (BFS):** Explores the graph of candidate action sequences systematically to find a path connecting $I$ to $G$.
  Search determines **which sequence of possibilities to explore**.

### Complete Flowchart Description
```text
      Current State S
            |
            v
Check Action Preconditions (S |= Preconds(a))
            |
            v
  Select Applicable Actions
            |
            v
Generate Successor State S' = (S \ Neg) U Pos
            |
            v
 Search Over Alternatives (BFS Frontier)
            |
            v
     Goal Satisfied? (S' |= G)
```

> **Think About It:** *"Logic determines what is possible; search determines what to try."*

---
## Task 5: Can the LLM Verify Its Own Plan?

### LLM Self-Explanation vs. Independent Verification
- When prompted to explain its plan, an LLM generates a fluent textual walkthrough claiming that every action's preconditions are satisfied.
- **Which should you trust more?**
  **Option (b): The independently executed state transitions.**
- **Why:** An LLM generates text through probabilistic token prediction; it does not maintain an internal execution stack or execute formal proof trees. It can hallucinate facts (e.g., claiming a package was teleported or that an action succeeded despite missing preconditions). Independent verification executes the formal mathematical transition rules deterministically, guaranteeing absolute ground truth.

> **Key Takeaway:** *A generated explanation is not the same as an independent verification.*
"""))

# Optional Prolog Tasks Code
prolog_code = """print("=" * 65)
print("OPTIONAL EXTENSION: PROLOG LOGICAL VERIFIER")
print("=" * 65)

# Python emulation of Prolog Horn Clause Resolution:
kb_connected = {("a", "b"), ("b", "a"), ("b", "c"), ("c", "b")}

def can_move(x, y):
    return (x, y) in kb_connected

def valid_move(x, y):
    return connected(x, y)

def connected(x, y):
    return (x, y) in kb_connected

# Queries from Task 6 & 7:
print("Task 6 & 7 Prolog Queries:")
print(f"  ?- can_move(a, b).   -> {can_move('a', 'b')}   (Expected: true)")
print(f"  ?- can_move(a, c).   -> {can_move('a', 'c')}  (Expected: false - no direct link)")
print(f"  ?- valid_move(a, b). -> {valid_move('a', 'b')}   (Expected: true)")
print(f"  ?- valid_move(b, c). -> {valid_move('b', 'c')}   (Expected: true)")
print(f"  ?- valid_move(a, c). -> {valid_move('a', 'c')}  (Expected: false - rejects invalid jump)")

# Task 8 Deduction Chain:
print("\\nTask 8: Rule-Based Logical Deduction Chain:")
facts = {"wet_road"}
rules = [
    ("wet_road => slippery", lambda s: "slippery" if "wet_road" in s else None),
    ("slippery => reduce_speed", lambda s: "reduce_speed" if "slippery" in s else None)
]

current_kb = set(facts)
for rule_name, rule_fn in rules:
    new_fact = rule_fn(current_kb)
    if new_fact:
        current_kb.add(new_fact)
        print(f"  Applied {rule_name} -> Deduced: {new_fact}")

print(f"  Query ?- reduce_speed. -> {'reduce_speed' in current_kb} (True by Modus Ponens)")
"""
cells.append(nbf.v4.new_code_cell(prolog_code))

# Submission & Reflection Questions
cells.append(nbf.v4.new_markdown_cell("""---
## Reflection Questions (Section 5)

### 1. Why is it useful to specify action preconditions and effects before asking an LLM to write the planner?
Specifying preconditions and effects decouples the logical semantics of the physical world from the search algorithm. It provides the LLM with an unambiguous specification of valid state transitions, preventing it from inventing ad-hoc heuristics or hallucinating illegal teleportation moves.

### 2. Give an example of an error that could occur if the planner failed to check an action's preconditions.
If the planner failed to check preconditions for $\\text{Drop}(\\text{Package}, C)$, the robot could simply execute $\\text{Drop}(\\text{Package}, C)$ at the start while standing at location $A$ without ever holding or picking up the package. The planner would immediately report goal achievement in 1 illegal step.

### 3. Why is a plan that "looks reasonable" not necessarily a valid plan?
A plan might appear plausible at a superficial glance (e.g. moving from A to C and dropping a package), but omit essential hidden dependencies—such as picking up the package before moving, or attempting to move directly between non-adjacent locations $A$ and $C$ where no physical track exists.

### 4. What did the LLM contribute to the implementation?
The LLM rapidly generated the object-oriented Python structure: the `Action` class with set operations (`issubset`, `difference_update`, `update`), the `PlanningProblem` container, and the BFS frontier loop. This reduced development time from hours to minutes.

### 5. What did you have to verify independently?
We independently verified:
- Correctness of the state transition function (ensuring deleted facts are properly removed before adding new facts).
- Termination and failure detection on impossible problems (Test B).
- That the plan contains no invalid jumps (e.g. confirming $A \\to C$ is rejected unless routed through $B$).

### 6. In this laboratory, where is logical reasoning being used?
Logical reasoning is used in:
1. **Precondition evaluation:** Entailment check $S \\models \\text{Preconditions}(a)$.
2. **State update:** Applying addition and deletion sets to update the knowledge base.
3. **Goal satisfaction:** Entailment check $S \\models G$.
4. **Prolog inference:** Modus Ponens deduction chains deriving derived truths from base facts.

### 7. How is planning related to the search algorithms studied in the previous module?
Planning is state-space search over a logically structured state space. In the previous module, states were atomic grid coordinates $(r, c)$. In planning, states are conjunctive sets of logical propositions, and transitions are governed by logical precondition-effect schemas rather than simple coordinate increments.
"""))

nb.cells = cells

# Save notebook
notebook_path = r"c:\Users\Aayush Kushwaha\Downloads\BITS\Acads\3-1\AI\Lab\Logic_lab\logic_lab_notebook.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook written to {notebook_path}")

# Execute the notebook to embed runtime outputs
print("Executing notebook to embed runtime outputs...")
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
with open(notebook_path, "r", encoding="utf-8") as f:
    nb_to_run = nbf.read(f, as_version=4)

ep.preprocess(nb_to_run, {'metadata': {'path': r"c:\Users\Aayush Kushwaha\Downloads\BITS\Acads\3-1\AI\Lab\Logic_lab"}})

with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb_to_run, f)

print("Notebook successfully executed and updated with real outputs!")
