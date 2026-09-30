# Laboratory Report – Logical Reasoning for Planning
## Using an LLM to Construct and Test a Simple Planning Agent

**Course:** CSF407 / Artificial Intelligence Laboratory  
**Topic:** STRIPS Planning, Preconditions and Effects, Logic + Search Integration, and Prolog Verification  
**Date:** September 30, 2026  
**Author:** AI Laboratory Student  

---

## 1. Executive Summary

This laboratory investigates the integration of **formal logical reasoning** with **state-space search** to construct autonomous planning agents:
$$\text{Logic} + \text{Search} = \text{Planning}$$

Using a classic warehouse robot logistics task with three locations $\{A, B, C\}$, an autonomous robot, and a package, we:
1. Specify the domain using formal STRIPS-style schemas with positive/negative preconditions and add/delete effects.
2. Formulate action applicability using logical entailment: $S \models \text{Preconditions}(a)$.
3. Manually trace a valid sequence of state transitions to establish ground-truth behavior.
4. Prompt an LLM to generate an automated Breadth-First Search (BFS) planner in Python.
5. Validate the planner across edge cases: a solvable baseline, an impossible problem without pick-up capabilities, and a goal with irrelevant movements.
6. Evaluate the reliability of LLM self-explanations versus independent state-transition verifiers.
7. Implement an executable rule-based verifier in **Prolog** to formally confirm movement validity and logical implication chains.

---

## 2. Task 0: Understand the Planning Problem

### 2.1 Formal Domain Specification
A planning problem is represented by the 3-tuple $\mathcal{P} = (I, A, G)$:
- **Initial State $I$:**
  $$I = \{\text{At}(\text{Robot}, A), \text{At}(\text{Package}, A)\}$$
- **Goal State $G$:**
  $$G = \{\text{At}(\text{Package}, C)\}$$
- **Available Actions $A$:**
  1. $\text{Move}(X, Y)$ between connected locations $(X, Y) \in \{(A, B), (B, A), (B, C), (C, B)\}$.
  2. $\text{PickUp}(\text{Package}, X)$ for $X \in \{A, B, C\}$.
  3. $\text{Drop}(\text{Package}, X)$ for $X \in \{A, B, C\}$.

### 2.2 Action Schemas: Preconditions and Effects

| Action | Preconditions | Positive Effects (Add List) | Negative Effects (Delete List) |
|:---|:---|:---|:---|
| $\text{Move}(X, Y)$ | $\text{At}(\text{Robot}, X)$ | $\text{At}(\text{Robot}, Y)$ | $\text{At}(\text{Robot}, X)$ |
| $\text{PickUp}(\text{Package}, X)$ | $\text{At}(\text{Robot}, X), \text{At}(\text{Package}, X)$ | $\text{Holding}(\text{Package})$ | $\text{At}(\text{Package}, X)$ |
| $\text{Drop}(\text{Package}, X)$ | $\text{At}(\text{Robot}, X), \text{Holding}(\text{Package})$ | $\text{At}(\text{Package}, X)$ | $\text{Holding}(\text{Package})$ |

### 2.3 Initial Action Applicability
Starting from $I = \{\text{At}(\text{Robot}, A), \text{At}(\text{Package}, A)\}$:
- **Is $\text{PickUp}(\text{Package}, A)$ applicable?**  
  **Yes.** Its preconditions are $\text{At}(\text{Robot}, A)$ and $\text{At}(\text{Package}, A)$. Both propositions are contained in $I$, satisfying $I \models \text{Preconditions}(\text{PickUp}(P, A))$.
- **Is $\text{Drop}(\text{Package}, C)$ applicable?**  
  **No.** Its preconditions are $\text{At}(\text{Robot}, C)$ and $\text{Holding}(\text{Package})$. Neither proposition is true in $I$.

> **Think About It:** An action cannot be executed merely because it is part of the catalog of available operators. It is applicable if and only if all its preconditions logically hold in the current state: $S \models \text{Preconditions}(a)$. Logical entailment is the gatekeeper of valid state transitions.

---

## 3. Task 1: Construct a Plan by Hand

A plan is an action sequence $\langle a_1, a_2, \dots, a_n \rangle$ producing valid state transitions:
$$I \xrightarrow{a_1} S_1 \xrightarrow{a_2} S_2 \dots \xrightarrow{a_n} S_n \quad \text{such that} \quad S_n \models G$$

### Manual State-Action Trace

| State | Action Applied | Resulting State Facts | Goal Satisfied? |
|:---:|:---|:---|:---:|
| **$S_0$** | *(Initial State)* | $\{\text{At}(\text{Robot}, A), \text{At}(\text{Package}, A)\}$ | No |
| **$S_1$** | $\text{PickUp}(\text{Package}, A)$ | $\{\text{At}(\text{Robot}, A), \text{Holding}(\text{Package})\}$ | No |
| **$S_2$** | $\text{Move}(A, B)$ | $\{\text{At}(\text{Robot}, B), \text{Holding}(\text{Package})\}$ | No |
| **$S_3$** | $\text{Move}(B, C)$ | $\{\text{At}(\text{Robot}, C), \text{Holding}(\text{Package})\}$ | No |
| **$S_4$** | $\text{Drop}(\text{Package}, C)$ | $\{\text{At}(\text{Robot}, C), \text{At}(\text{Package}, C)\}$ | **YES** ($S_4 \models G$) |

The optimal manual plan requires **4 actions**.

---

## 4. Task 2: Prompt Engineering & Python Implementation

### 4.1 Specification Prompt Provided to the LLM
```text
I want to implement a simple planning agent in Python.
Represent a state as a set of logical propositions.
Each action should contain:
- a name;
- positive preconditions;
- negative preconditions;
- positive effects;
- negative effects.

An action is applicable if all of its preconditions are satisfied by the current state.
When an action is applied:
1. remove its negative effects from the state;
2. add its positive effects to the state.

Use breadth-first search to find a sequence of actions that achieves a specified goal.
The program should also:
- detect when no plan exists;
- print the resulting sequence of actions;
- print the states reached after each action.

Explain the implementation and identify any assumptions you make.
```

### 4.2 Structural Code Inspection
- **Preconditions $\to$ Action Applicability:** Enforced via `self.pos_preconds.issubset(state) and self.neg_preconds.isdisjoint(state)`.
- **Effects $\to$ State Transition:** Computed as `new_state = (state - neg_effects) | pos_effects`.
- **Goal Test $\to$ Termination:** Evaluated when `self.goal_state.issubset(state)`.
- **BFS $\to$ Plan Exploration:** Explores candidate action sequences via a FIFO queue `collections.deque`, maintaining a `visited` set of frozensets to eliminate redundant cycles.

---

## 5. Task 3: Systematic Testing of the Generated Planner

We evaluated the planner across three deliberate experimental conditions:

| Test Case | Condition / Modification | Expected Outcome | Observed Result | Status |
|:---|:---|:---|:---|:---:|
| **Test A: Solvable Problem** | Original warehouse domain ($A \to B \to C$) | Finds optimal 4-step delivery plan | Discovered 4-step plan, expanded 6 states | **PASS** |
| **Test B: Impossible Problem** | Removed $\text{PickUp}$ action | Reports failure gracefully without inventing actions | Returned `None` ("No plan found"), expanded 3 states | **PASS** |
| **Test C: Irrelevant Actions** | Added autonomous robot moves without package | Does not confuse robot location with package location | Strictly satisfied $\text{At}(\text{Package}, C)$, ignoring irrelevant detours | **PASS** |

### Execution Trace for Test A:
1. **$S_0$:** `['At(Package, A)', 'At(Robot, A)']`
2. **Step 1:** Executed `PickUp(Package, A)` $\to S_1$: `['At(Robot, A)', 'Holding(Package)']`
3. **Step 2:** Executed `Move(A, B)` $\to S_2$: `['At(Robot, B)', 'Holding(Package)']`
4. **Step 3:** Executed `Move(B, C)` $\to S_3$: `['At(Robot, C)', 'Holding(Package)']`
5. **Step 4:** Executed `Drop(Package, C)` $\to S_4$: `['At(Package, C)', 'At(Robot, C)']`
- **Result:** Independently verified: **VALID**.

---

## 6. Task 4: Logic and Search Integration

### 6.1 Roles of Logic vs. Search
Planning combines two complementary ideas:
- **Logical Reasoning:** Operates *locally* on individual states and actions. Given state $S$ and action $a$, deductive reasoning tests whether $S \models \text{Preconditions}(a)$, and applies the add/delete lists to yield $S' = \text{Apply}(S, a)$.
  $$\text{Logic determines } \textbf{what is possible.}$$
- **State-Space Search (BFS):** Operates *globally* across the network of reachable states. It manages the frontier queue, tracks visited states, and chooses which candidate paths to extend.
  $$\text{Search determines } \textbf{what to try.}$$

### 6.2 The Complete Planning Loop
```text
           Current State S
                 |
                 v
     Check Action Preconditions
       (S |= Preconditions(a))
                 |
                 v
     Select Applicable Actions
                 |
                 v
      Generate Successor State
      S' = (S \ Neg) U Pos
                 |
                 v
      Search Over Alternatives
        (BFS Frontier Queue)
                 |
                 v
          Goal Satisfied?
            (S' |= G)
```

---

## 7. Task 5: Can the LLM Verify Its Own Plan?

### Question Analysis
When prompted to explain why its generated plan is valid, an LLM outputs a natural language narrative confirming that each precondition is satisfied.

**Which should you trust more:**
- **(a)** The LLM's explanation;
- **(b)** The independently executed state transitions?

**Answer:** **(b) The independently executed state transitions.**

**Justification:**
An LLM is a probabilistic autoregressive language model that predicts token sequences based on statistical patterns. It does not possess an internal logical solver or state-tracking execution environment. Consequently, an LLM can generate persuasive explanations that assert false preconditions are satisfied, hallucinate intermediate facts, or skip missing dependencies. In contrast, an independent transition engine evaluates the mathematical set operations deterministically according to formal logic.

> **Principle:** *A generated explanation is not the same as an independent verification.*

---

## 8. Optional Extension: Prolog as a Logical Verifier (Tasks 6, 7, 8)

### 8.1 Tasks 6 & 7: Warehouse Connectivity and Plan Verification
In `planner.pl`, the warehouse topology is encoded as Horn clauses:
```prolog
connected(a, b).
connected(b, a).
connected(b, c).
connected(c, b).

can_move(X, Y) :- connected(X, Y).
valid_move(X, Y) :- connected(X, Y).
```

**Query Results:**
1. `?- can_move(a, b).` $\implies$ **`true.`** (Direct fact `connected(a, b)` exists).
2. `?- can_move(a, c).` $\implies$ **`false.`** (Under the Closed-World Assumption, no direct link exists between $A$ and $C$).
3. `?- valid_move(a, b).` $\implies$ **`true.`**
4. `?- valid_move(b, c).` $\implies$ **`true.`**
5. `?- valid_move(a, c).` $\implies$ **`false.`** (Prolog successfully detects and rejects the invalid jump $A \to C$).

### 8.2 Task 8: Rule-Based Logical Deduction Chain
Consider the Prolog knowledge base:
```prolog
wet_road.
slippery :- wet_road.
reduce_speed :- slippery.
```
Query: `?- reduce_speed.` $\implies$ **`true.`**

**Logical Reasoning Chain (Modus Ponens):**
$$\text{Fact } (\text{wet\_road}) \implies \text{Rule } (\text{wet\_road} \to \text{slippery}) \implies \text{Rule } (\text{slippery} \to \text{reduce\_speed}) \implies \text{Conclusion } (\text{reduce\_speed})$$

### 8.3 Section 7.2 Reflection Questions
1. **Prolog Fact vs. Rule:** A *fact* is an unconditional assertion of truth (e.g. `connected(a, b).`), representing an atomic ground literal. A *rule* expresses a conditional assertion (e.g. `Head :- Body.`), meaning *Head is true if Body is true*.
2. **Prolog Query as Entailment:** A query `?- Goal.` asks whether the knowledge base $\mathcal{KB}$ logically entails the goal: $\mathcal{KB} \models \text{Goal}$. Prolog attempts to prove this using SLD resolution and backward chaining.
3. **Prolog for Plan Verification:** Python code can contain subtle indexing, state-mutation, or scoping bugs. Verifying plans in Prolog decouples plan generation from validation, checking proposed actions against an independent declarative knowledge base.
4. **Independent Verification for LLM Plans:** An LLM may propose plausible-sounding actions that violate physical constraints. An independent verifier provides an immutable ground-truth barrier that catches hallucinations before plans execute on physical robots.

---

## 9. Answers to Reflection Questions (Section 5)

### 1. Why is it useful to specify action preconditions and effects before asking an LLM to write the planner?
Specifying preconditions and effects provides a rigorous formal contract. It forces the developer to define exact state transition semantics, ensuring the LLM implements the intended domain physics rather than inventing ungrounded assumptions.

### 2. Give an example of an error that could occur if the planner failed to check an action's preconditions.
If preconditions were unchecked, the robot could execute $\text{Drop}(\text{Package}, C)$ while standing at location $A$ without ever picking up the package. The planner would report goal satisfaction in 1 illegal step.

### 3. Why is a plan that "looks reasonable" not necessarily a valid plan?
A plan might appear intuitive to human inspection (e.g. $\text{Move}(A, C) \to \text{Drop}(P, C)$), but violate implicit domain constraints—such as attempting to move between locations that share no physical corridor, or failing to pick up an item before transporting it.

### 4. What did the LLM contribute to the implementation?
The LLM accelerated software engineering by rapidly generating Python boilerplate: the object-oriented `Action` class, set-theoretic state manipulation methods, and the BFS search loop.

### 5. What did you have to verify independently?
We independently verified:
- Correct state update logic (ensuring negative effects are deleted before positive effects are added).
- Behavior on impossible and disconnected goals (confirming termination without infinite loops).
- Validity of each step in the generated plan against physical topology.

### 6. In this laboratory, where is logical reasoning being used?
Logical reasoning is used in:
- Precondition checking: evaluating logical entailment $S \models \text{Preconditions}(a)$.
- State transformation: computing truth assignments for successor states.
- Goal testing: verifying $S \models G$.
- Prolog verification: executing deductive resolution proofs over Horn clauses.

### 7. How is planning related to the search algorithms studied in the previous module?
Planning is state-space search over a structured logical representation. In the previous module, states were atomic coordinates $(r, c)$. In planning, states are combinatorial sets of logical relations, and transitions are governed by logical precondition-effect schemas, enabling rich multi-object reasoning.
